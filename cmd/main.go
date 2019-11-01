package main

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"time"

	"github.com/disintegration/imaging"

	"github.com/evanoberholster/face"
	"golang.org/x/image/draw"
)

func main() {
	det, err := face.NewFaceDetector("../models/mtcnn_1.14.pb")
	if err != nil {
		panic(err)
	}
	defer det.Close()

	buf, err := ioutil.ReadFile("../../test/img/12.jpg")
	if err != nil {
		panic(err)
	}
	// Reduce image to 2000px max edge

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
		//start := time.Now()
		f.AffineMatrix(256, 256)

		im := f.ToImage(i, draw.CatmullRom)
		//fmt.Println("Affine", time.Since(start))

		face.SaveJPG(fmt.Sprintf("p%d.jpg", idx), im, 80)
	}
	fmt.Println("Group", time.Since(startG))

	face.DrawDebugJPG("debug.jpg", i, faces)

	var buf2 bytes.Buffer
	i = imaging.Resize(i, 224, 244, imaging.NearestNeighbor)
	err = jpeg.Encode(&buf2, i, nil)
	if err != nil {
		panic(err)
	}
	// Aesthetics
	eval, err := face.NewAestheticsEvaluator("../models/nima_model.pb")
	if err != nil {
		panic(err)
	}

	defer eval.Close()
	aeImg, err := face.TensorFromJpeg(buf2.Bytes())
	if err != nil {
		fmt.Println(err)
	}

	//fmt.Println(aeImg.Shape())
	start = time.Now()
	score, _ := eval.Run(aeImg)
	fmt.Println(score)

	fmt.Println(time.Since(start))

	start = time.Now()
	fmt.Println(eval.Run(aeImg))
	fmt.Println(time.Since(start))

}
