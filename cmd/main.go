package main

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"time"

	"github.com/evanoberholster/face"
	"golang.org/x/image/draw"
)

func main() {
	det, err := face.NewFaceDetector("../models/mtcnn_rgb_1.14.pb")
	if err != nil {
		panic(err)
	}

	buf, err := ioutil.ReadFile("../../test/img/13.jpg")
	if err != nil {
		panic(err)
	}
	start := time.Now()
	tfImg, err := face.TensorFromJpeg(buf)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(time.Since(start))
	fmt.Println(tfImg.Shape())
	faces, err := det.DetectFaces(tfImg)
	fmt.Println(time.Since(start))
	fmt.Println(len(faces), faces)

	i, err := jpeg.Decode(bytes.NewReader(buf))
	if err != nil {
		panic(err)
	}
	startG := time.Now()

	for idx, f := range faces {
		start := time.Now()
		f.AffineMatrix(256, 256)
		fmt.Println("Affine", time.Since(start))
		im := f.ToImage(i, draw.BiLinear)

		face.SaveJPG(fmt.Sprintf("p%d.jpg", idx), im, 80)
	}
	fmt.Println("Group", time.Since(startG))

	face.DrawDebugJPG("debug.jpg", i, faces)
}
