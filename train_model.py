import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def create_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet',
                          include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    base_folder = os.getcwd()  # Get current working directory (ClassVision)
    target_size = (224, 224)
    batch_size = 32
    num_classes = 2  # Only two classes: lecture and no_lecture
    input_shape = (224, 224, 3)

    # Directories containing augmented images
    data_directories = [
        os.path.join(base_folder, 'lecture_augmented'),
        os.path.join(base_folder, 'no_lecture_augmented')
    ]

    # Create ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% of data for validation
    )

    # Training data generator
    train_generator = datagen.flow_from_directory(
        base_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=['lecture_augmented', 'no_lecture_augmented'],
        subset='training'
    )

    # Validation data generator
    validation_generator = datagen.flow_from_directory(
        base_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=['lecture_augmented', 'no_lecture_augmented'],
        subset='validation'
    )

    model = create_model(input_shape, num_classes)

    # Ensure directories exist for checkpoints and model saving
    os.makedirs('./model/checkpoint', exist_ok=True)
    os.makedirs('./model/saved_model', exist_ok=True)

    # Define callbacks
    checkpoint = ModelCheckpoint(filepath='./model/checkpoint/model-{epoch:02d}-{val_accuracy:.2f}.keras',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max')
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=5, mode='max')

    # Train the model
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )

    # Save the final model
    model.save('./model/saved_model/final_model.keras')  # Save as .keras


if __name__ == "__main__":
    main()
