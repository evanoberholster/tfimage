package main

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"time"

	"github.com/evanoberholster/gg"
	"golang.org/x/image/draw"
)

func main() {
	det, err := NewMtcnnDetector("models/mtcnn_rgb_1.14.pb")
	if err != nil {
		panic(err)
	}

	buf, err := ioutil.ReadFile("../test/img/13.jpg")
	if err != nil {
		fmt.Println(err)
	}
	start := time.Now()
	tfImg, err := TensorFromJpeg(buf)
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
	ctx := gg.NewContextForImage(i)
	for idx, f := range faces {
		// Draw Face
		ctx.Push()
		ctx.DrawRectangle(float64(f.Bbox[1]), float64(f.Bbox[0]), float64(f.Bbox[3]-f.Bbox[1]), float64(f.Bbox[2]-f.Bbox[0]))
		ctx.SetRGBA(150, 0, 0, 0.3)
		ctx.Fill()
		ctx.Pop()

		// Draw Spots
		ctx.Push()
		x, y := f.LeftEye()
		ctx.DrawPoint(x, y, 10)
		x, y = f.RightEye()
		ctx.DrawPoint(x, y, 10)
		x, y = f.LeftMouth()
		ctx.DrawPoint(x, y, 10)
		x, y = f.RightMouth()
		ctx.DrawPoint(x, y, 10)
		x, y = f.Nose()
		ctx.DrawPoint(x, y, 10)
		x, y = f.EyesCenter()
		ctx.DrawPoint(x, y, 10)
		ctx.SetRGBA(0, 150, 0, 0.7)
		ctx.Fill()
		ctx.Pop()

		start := time.Now()
		f.AffineMatrix(256, 256, i)
		im := f.ToImage(i, draw.CatmullRom)
		fmt.Println(time.Since(start))
		SaveJPG(fmt.Sprintf("out1b%d.jpg", idx), im, 80)

	}
	fmt.Println("Group", time.Since(startG))

	ctx.SaveJPG("out.jpg", 80)

}
