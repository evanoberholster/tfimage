package tfimage

import (
	"image"
	"image/jpeg"
	"os"

	"github.com/evanoberholster/gg"
)

// SaveJPG - Save image to JPG
func SaveJPG(path string, im image.Image, quality int) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return jpeg.Encode(file, im, &jpeg.Options{Quality: quality})
}

func (fr *FaceResults) DrawDebugJPEG(path string, src image.Image) error {
	ctx := gg.NewContextForImage(src)
	for _, f := range fr.results {
		// Draw Face
		ctx.Push()
		ctx.DrawRectangle(float64(f.box[1]), float64(f.box[0]), float64(f.box[3]-f.box[1]), float64(f.box[2]-f.box[0]))
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
	}
	return SaveJPG(path, ctx.Image(), 70)
}
