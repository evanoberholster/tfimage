package tfimage

import (
	"fmt"
	"image"
	"io/ioutil"
	"math"
	"strings"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"golang.org/x/image/draw"
)

// Errors
const (
	ErrLoadFile string = "Unable to Load MTCNN model file: %v\n"
)

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
	if options.MinimumSize < 30 {
		options.MinimumSize = 30 // default size
	}
	if options.FaceHeight == 0 || options.FaceWidth == 0 {
		options.FaceWidth = 256
		options.FaceHeight = 256
	}
	det := &FaceDetector{scaleFactor: 0.709, scoreThresholds: []float32{0.6, 0.7, 0.8}, Options: options}
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, fmt.Errorf(ErrLoadFile, modelFile)
	}

	det.graph = tf.NewGraph()
	if err := det.graph.Import(model, ""); err != nil {
		return nil, err
	}

	det.session, err = tf.NewSession(det.graph, nil)
	if err != nil {
		return nil, err
	}

	return det, nil
}

// Close - Close a FaceDetector Session
func (det *FaceDetector) Close() {
	if det.session != nil {
		det.session.Close()
		det.session = nil
	}
}

//
//
//

// DetectFaces runs the tensorflow detection session and outputs a FacesResults
func (det *FaceDetector) DetectFaces(tensor *tf.Tensor) (*FaceResults, error) {
	start := time.Now()
	minSize, err := tf.NewTensor(float32(det.Options.MinimumSize))
	if err != nil {
		return nil, fmt.Errorf("error minimum size: %v", err)
	}
	threshold, err := tf.NewTensor(det.scoreThresholds)
	if err != nil {
		return nil, fmt.Errorf("error score threshold: %v", err)
	}
	factor, err := tf.NewTensor(det.scaleFactor)
	if err != nil {
		return nil, fmt.Errorf("error scale factor: %v", err)
	}

	graph := det.graph
	output, err := det.session.Run(
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
		return nil, fmt.Errorf("error tensorflow Face Detection: %v", err)
	}

	res := &FaceResults{}

	if len(output) > 0 {
		prob := output[0].Value().([]float32)
		landmarks := output[1].Value().([][]float32)
		bbox := output[2].Value().([][]float32)

		res.results = make([]Face, len(prob))
		for i := 0; i < len(prob); i++ {
			res.results[i] = newFace(prob[i], bbox[i], landmarks[i])
		}
	}
	res.d = time.Since(start)
	return res, nil
}

type FaceResults struct {
	results []Face
	d       time.Duration
}

func (fr FaceResults) String() string {
	sb := strings.Builder{}
	sb.WriteString(fmt.Sprintf("%d Faces detected.\t in %s \n", fr.Len(), fr.d))
	for _, r := range fr.results {
		sb.WriteString(r.String())
	}
	return sb.String()
}

func (fr FaceResults) Len() int {
	return len(fr.results)
}

func (fr FaceResults) ToJPEG(src image.Image, kernel draw.Interpolator, width uint16, height uint16, fn func(image.Image) error) (err error) {
	faceImage := image.NewRGBA(image.Rect(0, 0, int(width), int(height)))
	for _, f := range fr.results {
		m := f.AffineMatrix(width, height).ToAffineMatrix()
		kernel.Transform(faceImage, m, src, src.Bounds(), draw.Src, nil)
		if err = fn(faceImage); err != nil {
			return err
		}
	}
	return nil
}

//
//
//

// newFace creates a new face with probability, bounding box, and landmarks
func newFace(probablity float32, bbox []float32, landmarks []float32) Face {
	fr := Face{p: probablity}
	copy(fr.landmarks[:], landmarks)
	copy(fr.box[:], bbox)
	return fr
}

// Face
type Face struct {
	box       [4]float32
	landmarks [10]float32
	p         float32
}

func (f Face) String() string {
	w, h := f.Size()
	return fmt.Sprintf(" Probability: %.2f%% \t Size: %dx%d \t Angle: %.4f \n", f.p*100, w, h, f.Angle())
}

// Size returns the width and height of the face in the source image
func (f Face) Size() (width int, height int) {
	width = int(f.box[3]) - int(f.box[1])
	height = int(f.box[2]) - int(f.box[0])
	return
}

// RightEye returns the (x,y) of the right eye
func (f Face) RightEye() (x float64, y float64) {
	x = float64(f.landmarks[6])
	y = float64(f.landmarks[1])
	return
}

// LeftEye returns the (x,y) of the left eye
func (f Face) LeftEye() (x float64, y float64) {
	x = float64(f.landmarks[5])
	y = float64(f.landmarks[0])
	return
}

// RightMouth returns the (x,y) of the right corner of the mouth
func (f Face) RightMouth() (x float64, y float64) {
	x = float64(f.landmarks[7])
	y = float64(f.landmarks[2])
	return
}

// LeftMouth returns the (x,y) of the left corner of the mouth
func (f Face) LeftMouth() (x float64, y float64) {
	x = float64(f.landmarks[8])
	y = float64(f.landmarks[3])
	return
}

// Nose returns the (x,y) center of the nose
func (f Face) Nose() (x float64, y float64) {
	x = float64(f.landmarks[9])
	y = float64(f.landmarks[4])
	return
}

// EyesCenter returns the (x,y) center between the eyes
func (f Face) EyesCenter() (x float64, y float64) {
	x1, y1 := f.LeftEye()
	x2, y2 := f.RightEye()
	return (x1 + x2) / 2, (y1 + y2) / 2
}

// Angle caculates the angle of the face using the LeftEye and RightEye
// returns radians
func (f Face) Angle() float64 {
	x0, y0 := f.LeftEye()
	x1, y1 := f.RightEye()
	return math.Atan2(y0-y1, x0-x1)
}

//AffineMatrix builds a Face Warp Affine Matrix
func (f *Face) AffineMatrix(width, height uint16) Matrix {
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
	desiredDist *= float64(width)

	// Set the translation Matrix with the desired positions of the eyes
	tX := float64(width) * 0.5
	tY := float64(height) * desiredLeftEyeY

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

	matrix := RotationMatrix2D(eyesX, eyesY, angle, scale)

	// Adjust position of the image
	matrix.AdjustPosition((tX - eyesX), (tY - eyesY))

	// unCenter the matrix from around the eyes
	//f.matrix = f.matrix.Translate(-eyesX, -eyesY)
	return matrix
}

// ToImage transforms an image by the matrix and returns the face image
func (f Face) ToImage(srcImage image.Image, kernel draw.Interpolator, width uint16, height uint16) image.Image {
	faceImg := image.NewRGBA(image.Rect(0, 0, int(width), int(height)))
	s2d := f.AffineMatrix(width, height).ToAffineMatrix()
	kernel.Transform(faceImg, s2d, srcImage, srcImage.Bounds(), draw.Src, nil)
	return faceImg
}
