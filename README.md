# ClassVision

ClassVision is a machine learning project aimed at predicting whether a lecture is currently in progress or not. By analyzing various audio and visual cues, the system can provide real-time insights into the classroom environment.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In today's digital age, remote learning has become increasingly prevalent. However, it can be challenging for students to determine whether a lecture is actively taking place or if it's a pre-recorded session. ClassVision aims to address this issue by leveraging machine learning techniques to predict the presence of a live lecture.

## Installation

To use ClassVision, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/ClassVision.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Access the application by opening the following URL in your web browser:

```
http://localhost:5000
```

## Usage

Once the application is running, you can upload an image on the frontend to predict whether a lecture is in progress or not.

If you want to retrain the model, follow these additional steps:

1. Add data to the `lecture` and `no_lecture` folders.
2. Run the training script:

```bash
python train_model.py
```

3. After training, run the application again:

```bash
python app.py
```

## Contributing

Contributions are welcome! If you would like to contribute to ClassVision, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License

ClassVision is licensed under the [MIT License](LICENSE).
