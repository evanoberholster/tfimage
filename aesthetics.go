package tfimage

import (
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
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
}

// NewAestheticsEvaluator - Creates a new Aesthetics Evaluator
func NewAestheticsEvaluator(modelFile string) (*AestheticsEvaluator, error) {
	eval := &AestheticsEvaluator{}

	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, err
	}

	eval.graph = tf.NewGraph()
	if err := eval.graph.Import(model, ""); err != nil {
		return nil, err
	}

	eval.session, err = tf.NewSession(eval.graph, nil)
	if err != nil {
		return nil, err
	}
	return eval, nil
}

// Close closes the Aesthetics Evaluator's Session
func (eval *AestheticsEvaluator) Close() {
	if eval.session != nil {
		eval.session.Close()
		eval.graph = nil
		eval.session = nil
	}
}

func (eval *AestheticsEvaluator) Run(tensor *tf.Tensor) (score float32, err error) {

	graph := eval.graph
	output, err := eval.session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input_1").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("dense_1/Softmax").Output(0),
		},
		nil,
	)
	if err != nil {
		return 0, err
	}

	if len(output) > 0 {
		values := output[0].Value().([][]float32)[0]
		score = float32(calcScore(values))
		//fmt.Println(output[0].Shape())
	}
	return score, nil
}

func calcScore(values []float32) (sum float32) {
	var tempSum float32
	// Sum
	for _, v := range values {
		tempSum += v
	}
	// Normalize values
	for idx, v := range values {
		sum += (v / tempSum) * (float32(idx) + 1)
	}
	return sum
}
