package face

import (
	"fmt"
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Detector - Detector for faces in an image based on an MTCNN model.
// Uses a Multi-task Cascaded Convolutional Network Detector trained from:
// https://kpzhang93.github.io/MTCNN_face_detection_alignment/
type Detector struct {
	graph   *tf.Graph
	session *tf.Session

	minSize         float32
	scaleFactor     float32
	scoreThresholds []float32
}

// NewFaceDetector - Create a New FaceDetector from a model file
func NewFaceDetector(modelFile string) (*Detector, error) {
	det := &Detector{minSize: 30.0, scaleFactor: 0.709, scoreThresholds: []float32{0.6, 0.7, 0.8}}
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
func (det *Detector) Config(scaleFactor, minSize float32, scoreThresholds []float32) {
	if scaleFactor > 0 {
		det.scaleFactor = scaleFactor
	}
	if minSize > 0 {
		det.minSize = minSize
	}
	if scoreThresholds != nil {
		det.scoreThresholds = scoreThresholds
	}
}

// Close - Close a FaceDetector Session
func (det *Detector) Close() {
	if det.session != nil {
		det.session.Close()
		det.session = nil
	}
}

func runScope(s *op.Scope, inputs map[tf.Output]*tf.Tensor, outputs []tf.Output) ([]*tf.Tensor, error) {
	graph, err := s.Finalize()
	if err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	return session.Run(inputs, outputs, nil)
}

// TensorFromJpeg - Decode a JPEG image into RGB channels in a tensor
func TensorFromJpeg(bytes []byte) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}

	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	out := op.ExpandDims(s,
		op.Cast(s, op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
		op.Const(s.SubScope("make_batch"), int32(0)))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{input: tensor}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

// DetectFaces - runs the tensorflow detection session and outputs an array of Faces
func (det *Detector) DetectFaces(tensor *tf.Tensor) ([]Face, error) {
	session := det.session
	graph := det.graph

	minSize, err := tf.NewTensor(det.minSize)
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
		fmt.Println(err)
		return nil, err
	}
	var faces []Face
	if len(output) > 0 {
		prob := output[0].Value().([]float32)
		landmarks := output[1].Value().([][]float32)
		bbox := output[2].Value().([][]float32)
		for idx := range prob {
			faces = append(faces, Face{
				Bbox:      bbox[idx],
				Prob:      prob[idx],
				Landmarks: landmarks[idx],
			})
		}
	}
	return faces, nil
}
