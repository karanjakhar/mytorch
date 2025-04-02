# MyTorch

MyTorch is a neural network implementation from scratch in the C programming language. This project demonstrates the fundamentals of machine learning by training a neural network on the MNIST digit dataset.

## Features

- Fully implemented in C for educational purposes.
- Supports training and inference on the MNIST dataset.
- Lightweight and dependency-free.

## Requirements

- A C compiler (e.g., GCC or Clang).
- MNIST dataset files (can be downloaded from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)).

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mytorch.git
    cd mytorch
    ```

2. Compile the code:
    ```bash
    gcc -o mytorch main.c read_mnist.c -lm
    ```

3. Run the program:
    ```bash
    ./mytorch
    ```

## Project Structure

- `main.c`: Entry point of the program.
- `README.md`: Project documentation.

## Dataset

Ensure the MNIST dataset files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, etc.) are available in the working directory. The program will load and preprocess the data automatically.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MNIST dataset by Yann LeCun et al.
- Inspiration from various neural network implementations.
