package main

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"time"

	"github.com/disintegration/imaging"
	"github.com/evanoberholster/tfimage"
	"golang.org/x/image/draw"
)

func main() {
	opts := tfimage.FaceDetectorOptions{
		MinimumSize: 100, // 50 Pixels
	}
	det, err := tfimage.NewFaceDetector("../models/mtcnn_1.14.pb", opts)
	if err != nil {
		panic(err)
	}
	defer det.Close()

	buf, err := ioutil.ReadFile("../../test/img/13.jpg")
	if err != nil {
		panic(err)
	}
	// Reduce image to 2000px max edge

	start := time.Now()
	tfImg, err := tfimage.TensorFromJpeg(buf)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Time to Tensor: ", time.Since(start))

	fmt.Println(tfImg.Shape())
	startd := time.Now()
	faces, err := det.DetectFaces(tfImg)
	fmt.Println("Time to detect faces #1:", time.Since(startd))

	startd = time.Now()
	faces, err = det.DetectFaces(tfImg)
	fmt.Println("Time to detect faces #2:", time.Since(startd))

	fmt.Println(len(faces), faces)

	i, err := jpeg.Decode(bytes.NewReader(buf))
	if err != nil {
		panic(err)
	}
	startG := time.Now()

	for idx, f := range faces {
		//start := time.Now()
		f.AffineMatrix()

		im := f.ToImage(i, draw.CatmullRom)
		//fmt.Println("Affine", time.Since(start))

		tfimage.SaveJPG(fmt.Sprintf("p%d.jpg", idx), im, 80)
	}
	fmt.Println("Group", time.Since(startG))

	tfimage.DrawDebugJPG("debug.jpg", i, faces)

	var buf2 bytes.Buffer
	i = imaging.Resize(i, 800, 600, imaging.CatmullRom)
	err = jpeg.Encode(&buf2, i, &jpeg.Options{Quality: 90})
	if err != nil {
		panic(err)
	}
	// Aesthetics
	eval, err := tfimage.NewAestheticsEvaluator("../models/nima_1.14.pb")
	if err != nil {
		panic(err)
	}

	defer eval.Close()
	aeImg, err := tfimage.TensorFromJpeg(buf2.Bytes())
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(time.Since(start))

	start = time.Now()
	fmt.Println(eval.Run(aeImg))
	fmt.Println(time.Since(start))

}
