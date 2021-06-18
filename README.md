# tfImage

Work in progress
- Face detection (MTCNN)
- Image aesthetics (NIMA)

## Example Code
 See cmd/main.go

## Install Tensorflow v1.14
Warning: This tensorflow binary is CPU only and doesnt support AVX2 and FMA

```sudo curl -L https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz | sudo tar xz --directory /usr/local ```
``` sudo ldconfig ```

``` go get github.com/tensorflow/tensorflow/tensorflow/go@v1.14.0 ```


