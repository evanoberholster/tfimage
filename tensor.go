package tfimage

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

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
