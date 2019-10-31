package face

import (
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"
	"golang.org/x/image/math/f64"
)

// Face -
type Face struct {
	Prob          float32
	Bbox          []float32
	Landmarks     []float32
	matrix        Matrix
	dstFaceHeight int
	dstFaceWidth  int
}

// NewFace creates a new face with a probability
func NewFace(p float32) Face {
	return Face{
		Prob:   p,
		matrix: NewMatrix(),
	}
}

func (f Face) String() string {
	return fmt.Sprintf("Probability: %.4f \t Angle: %.4f\n", f.Prob, f.Angle())
}

// RightEye -
func (f Face) RightEye() (x float64, y float64) {
	x = float64(f.Landmarks[6])
	y = float64(f.Landmarks[1])
	return
}

// LeftEye -
func (f Face) LeftEye() (x float64, y float64) {
	x = float64(f.Landmarks[5])
	y = float64(f.Landmarks[0])
	return
}

// RightMouth -
func (f Face) RightMouth() (x float64, y float64) {
	x = float64(f.Landmarks[7])
	y = float64(f.Landmarks[2])
	return
}

// LeftMouth -
func (f Face) LeftMouth() (x float64, y float64) {
	x = float64(f.Landmarks[8])
	y = float64(f.Landmarks[3])
	return
}

// Nose -
func (f Face) Nose() (x float64, y float64) {
	x = float64(f.Landmarks[9])
	y = float64(f.Landmarks[4])
	return
}

// EyesCenter - Return the center between the eyes
func (f Face) EyesCenter() (x float64, y float64) {
	x1, y1 := f.LeftEye()
	x2, y2 := f.RightEye()
	return (x1 + x2) / 2, (y1 + y2) / 2
}

// Angle - Caculates the angle of the face using the LeftEye and RightEye
// returns radians
func (f Face) Angle() float64 {
	x0, y0 := f.LeftEye()
	x1, y1 := f.RightEye()
	return math.Atan2(y0-y1, x0-x1)
}

// ToImage transforms an image by the matrix and returns the face image
func (f Face) ToImage(im image.Image, kernel draw.Interpolator) *image.RGBA {
	faceImg := image.NewRGBA(image.Rect(0, 0, f.dstFaceWidth, f.dstFaceHeight))
	s2d := f64.Aff3{f.matrix.XX, f.matrix.XY, f.matrix.X0, f.matrix.YX, f.matrix.YY, f.matrix.Y0}
	kernel.Transform(faceImg, s2d, im, im.Bounds(), draw.Src, nil)
	return faceImg
}

//AffineMatrix - Face Warp Affine Matrix
func (f *Face) AffineMatrix(faceWidth, faceHeight int) {
	// Output Size
	f.dstFaceWidth, f.dstFaceHeight = faceWidth, faceHeight
	desiredLeftEyeX, desiredLeftEyeY := 0.33, 0.30

	// Eye Points
	x1, y1 := f.LeftEye()
	x2, y2 := f.RightEye()
	dY := y2 - y1
	dX := x2 - x1

	// Caclulate the angle between the eye points
	angle := math.Atan2(dY, dX)

	// Calculate the ecludian distance between eye points
	dist := math.Sqrt((dX * dX) + (dY * dY))

	// Determine the scale of the resulting image by taking the ratio
	// of the distance between the eyes in the image and ratio of the
	// distance between the eyes in the output image
	desiredRightEyeX := 1.0 - desiredLeftEyeX
	desiredDist := (desiredRightEyeX - desiredLeftEyeX)
	desiredDist *= float64(f.dstFaceWidth)

	// Set the translation Matrix with the desired positions of the eyes
	tX := float64(f.dstFaceWidth) * 0.5
	tY := float64(f.dstFaceHeight) * desiredLeftEyeY

	// Caclulate the scale of the face
	scale := desiredDist / dist

	// Calculate the mean distance between the eyes
	eyesX, eyesY := f.EyesCenter()

	// Center the matrix around the eyes
	//f.matrix = f.matrix.Translate(eyesX, eyesY)

	// Adjust the scale of the matrix
	//f.matrix = f.matrix.AdjustScale(scale)

	// Rotate Matrix in the opposite direction of the angle of the eyes
	//f.matrix = f.matrix.Rotate(angle * -1)

	f.matrix = RotationMatrix2D(eyesX, eyesY, angle, scale)

	// Adjust position of the image
	f.matrix.AdjustPosition((tX - eyesX), (tY - eyesY))

	// unCenter the matrix from around the eyes
	//f.matrix = f.matrix.Translate(-eyesX, -eyesY)
}
