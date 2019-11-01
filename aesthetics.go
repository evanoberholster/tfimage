package ifimage

import (
	"fmt"
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

//input_1 (InputLayer)         (None, 224, 224, 3)       0
// MobileNet

// Evaluator - Evaluator for aesthetic image quality.
// Uses a MobileNet modified CNN trained from:
// https://github.com/idealo/image-quality-assessment/
// Apache 2.0 License
type AestheticsEvaluator struct {
	graph   *tf.Graph
	session *tf.Session

	//minSize         float32
	//scaleFactor     float32
	//scoreThresholds []float32
}

// NewAestheticsEvaluator - Creates a new Aesthetics Evaluator
func NewAestheticsEvaluator(modelFile string) (*AestheticsEvaluator, error) {
	eval := &AestheticsEvaluator{}
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

	eval.graph = graph
	eval.session = session

	return eval, nil
}

// Close - Close a AestheticsEvaluator Session
func (eval *AestheticsEvaluator) Close() {
	if eval.session != nil {
		eval.session.Close()
		eval.session = nil
	}
}

func (eval *AestheticsEvaluator) Run(tensor *tf.Tensor) (float64, error) {
	session := eval.session
	graph := eval.graph

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input_1").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("dense_1/Softmax").Output(0),
		},
		nil,
	)
	if err != nil {
		fmt.Println(err)
		panic(err)
		return 0, err
	}
	score := 0.0
	if len(output) > 0 {
		values := output[0].Value().([][]float32)[0]
		score = calcScore(values)
		//fmt.Println(output[0].Shape())
	}
	return score, nil
}

func calcScore(values []float32) float64 {
	var sum float64
	scores := make([]float64, len(values))
	// Sum
	for _, v := range values {
		sum += float64(v)
	}
	// Normalize
	for idx, v := range values {
		scores[idx] = (float64(v) / sum) * (float64(idx) + 1)
	}
	// Sum
	sum = 0
	for _, v := range scores {
		sum += v
	}
	return sum
}

// AestheticsFromJpeg - Decode a JPEG image into RGB channels in a tensor
func AestheticsFromJpeg(bytes []byte) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}

	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	out := op.Cast(s, op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float)
	//out := op.ExpandDims(s,
	//	op.Cast(s, op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
	//	op.Const(s.SubScope("make_batch"), int32(0)))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{input: tensor}, []tf.Output{out})
	if err != nil {
		fmt.Println(s.Err())
		return nil, err
	}

	return outs[0], nil
}
