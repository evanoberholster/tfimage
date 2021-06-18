package main

import (
	"bytes"
	"fmt"
	"image"
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

	buf, err := ioutil.ReadFile("../../test/img/f.jpg")
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

	faceResults, err := det.DetectFaces(tfImg)
	if err != nil {
		panic(err)
	}
	fmt.Println(faceResults)

	srcImage, err := jpeg.Decode(bytes.NewReader(buf))
	if err != nil {
		panic(err)
	}
	start = time.Now()
	n := 0
	faceResults.ToJPEG(srcImage, draw.CatmullRom, 256, 256, func(faceImage image.Image) error {
		n++
		return tfimage.SaveJPG(fmt.Sprintf("p%d.jpg", n), faceImage, 80)
	})
	fmt.Println("Time taken to save images to disk:", time.Since(start))
	faceResults.DrawDebugJPEG("debug.jpg", srcImage)

	var buf2 bytes.Buffer
	srcImage = imaging.Resize(srcImage, 800, 600, imaging.CatmullRom)
	err = jpeg.Encode(&buf2, srcImage, &jpeg.Options{Quality: 90})
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

	start = time.Now()
	fmt.Println(eval.Run(aeImg))
	fmt.Println("Time to calculate visual asthetic of image: ", time.Since(start))
}
