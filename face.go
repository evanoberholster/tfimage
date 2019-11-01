package tfimage

import (
	"fmt"
	"image"
	"io/ioutil"
	"math"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
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

// FaceDetector - Detector for faces in an image based on an MTCNN model.
// Uses a Multi-task Cascaded Convolutional Network Detector trained from:
// https://kpzhang93.github.io/MTCNN_face_detection_alignment/
type FaceDetector struct {
	graph   *tf.Graph
	session *tf.Session

	Options         FaceDetectorOptions
	scaleFactor     float32
	scoreThresholds []float32
}

// FaceDetectorOptions -
type FaceDetectorOptions struct {
	MinimumSize int
	FaceWidth   int
	FaceHeight  int
}

// NewFaceDetector - Create a New FaceDetector from a model file
func NewFaceDetector(modelFile string, options FaceDetectorOptions) (*FaceDetector, error) {
	det := &FaceDetector{scaleFactor: 0.709, scoreThresholds: []float32{0.6, 0.7, 0.8}, Options: options}
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, err
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	det.graph = graph
	det.session = session

	return det, nil
}

// Config - Set options for scaleFactor, minSize and scoreThresholds
func (det *FaceDetector) Config(scaleFactor, minSize float32, scoreThresholds []float32) {
	if scaleFactor > 0 {
		det.scaleFactor = scaleFactor
	}
	if scoreThresholds != nil {
		det.scoreThresholds = scoreThresholds
	}
}

// Close - Close a FaceDetector Session
func (det *FaceDetector) Close() {
	if det.session != nil {
		det.session.Close()
		det.session = nil
	}
}

// DetectFaces - runs the tensorflow detection session and outputs an array of Faces
func (det *FaceDetector) DetectFaces(tensor *tf.Tensor) ([]Face, error) {
	session := det.session
	graph := det.graph

	if det.Options.MinimumSize == 0 {
		det.Options.MinimumSize = 30 // default size
	}
	if det.Options.FaceHeight == 0 || det.Options.FaceWidth == 0 {
		det.Options.FaceWidth = 256
		det.Options.FaceHeight = 256
	}

	minSize, err := tf.NewTensor(float32(det.Options.MinimumSize))
	if err != nil {
		return nil, fmt.Errorf("Minimum Size Error: %v", err)
	}
	threshold, err := tf.NewTensor(det.scoreThresholds)
	if err != nil {
		return nil, fmt.Errorf("Threshold Error: %v", err)
	}
	factor, err := tf.NewTensor(det.scaleFactor)
	if err != nil {
		return nil, fmt.Errorf("Scale Factor Error: %v", err)
	}

	var faces []Face

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("sub").Output(0):        tensor,
			graph.Operation("min_size").Output(0):   minSize,
			graph.Operation("thresholds").Output(0): threshold,
			graph.Operation("factor").Output(0):     factor,
		},
		[]tf.Output{
			graph.Operation("prob").Output(0),
			graph.Operation("landmarks").Output(0),
			graph.Operation("box").Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(output) > 0 {
		prob := output[0].Value().([]float32)
		landmarks := output[1].Value().([][]float32)
		bbox := output[2].Value().([][]float32)
		for idx := range prob {
			faces = append(faces, NewFace(prob[idx], bbox[idx], landmarks[idx], det.Options.FaceWidth, det.Options.FaceHeight))
		}
	}
	return faces, nil
}

// NewFace creates a new face with a probability
func NewFace(probability float32, bbox []float32, landmarks []float32, dstWidth, dstHeight int) Face {
	return Face{
		Prob:          probability,
		Bbox:          bbox,
		Landmarks:     landmarks,
		matrix:        NewMatrix(),
		dstFaceHeight: dstHeight,
		dstFaceWidth:  dstWidth,
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
func (f *Face) AffineMatrix() {
	// Output Size
	//f.dstFaceWidth, f.dstFaceHeight = faceWidth, faceHeight
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
