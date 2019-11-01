package tfimage

// Modified from github.com/fogleman/gg
// Oct 28, 2019
// Michael Fogleman and Contributors
// LICENSE: MIT

import "math"

// Matrix -
type Matrix struct {
	XX, YX, XY, YY, X0, Y0 float64
}

// NewMatrix - Create a new Matrix
func NewMatrix() Matrix {
	return Matrix{
		1, 0,
		0, 1,
		0, 0,
	}
}

// Translate New Matrix by x and y
func Translate(x, y float64) Matrix {
	return Matrix{
		1, 0,
		0, 1,
		x, y,
	}
}

// Scale New Matrix by x and y
func Scale(x, y float64) Matrix {
	return Matrix{
		x, 0,
		0, y,
		0, 0,
	}
}

// Rotate New Matrix by x and y
func Rotate(angle float64) Matrix {
	c := math.Cos(angle)
	s := math.Sin(angle)
	return Matrix{
		c, s,
		-s, c,
		0, 0,
	}
}

// Shear New Matrix by x and y
func Shear(x, y float64) Matrix {
	return Matrix{
		1, y,
		x, 1,
		0, 0,
	}
}

// Multiply Matrix and another Matrix
func (m Matrix) Multiply(b Matrix) Matrix {
	return Matrix{
		m.XX*b.XX + m.YX*b.XY,
		m.XX*b.YX + m.YX*b.YY,
		m.XY*b.XX + m.YY*b.XY,
		m.XY*b.YX + m.YY*b.YY,
		m.X0*b.XX + m.Y0*b.XY + b.X0,
		m.X0*b.YX + m.Y0*b.YY + b.Y0,
	}
}

// TransformVector using a Matrix and x and y
func (m Matrix) TransformVector(x, y float64) (tx, ty float64) {
	tx = m.XX*x + m.XY*y
	ty = m.YX*x + m.YY*y
	return
}

// TransformPoint using a Matrix and x and y
func (m Matrix) TransformPoint(x, y float64) (tx, ty float64) {
	tx = m.XX*x + m.XY*y + m.X0
	ty = m.YX*x + m.YY*y + m.Y0
	return
}

// Translate a Matrix using x and y
func (m Matrix) Translate(x, y float64) Matrix {
	return Translate(x, y).Multiply(m)
}

// Scale a Matrix using x and y
func (m Matrix) Scale(x, y float64) Matrix {
	return Scale(x, y).Multiply(m)
}

// AdjustScale -
func (m Matrix) AdjustScale(scale float64) Matrix {
	m.XX *= scale
	m.XY *= scale
	m.YX *= scale
	m.YY *= scale
	return m
}

// RotationMatrix2D creates a Rotation and Transformation Matrix.
// This is similiar to OpenCV's function.
// Angle in Radians
func RotationMatrix2D(centerX, centerY float64, angle, scale float64) Matrix {
	a := scale * math.Cos(angle)
	b := scale * math.Sin(angle)
	return Matrix{
		a, -b,
		b, a,
		(1-a)*centerX - b*centerY, b*centerX + (1-a)*centerY,
	}
}

// AdjustPosition adjust position of the matrix by x and y
func (m *Matrix) AdjustPosition(x, y float64) {
	m.X0 += x
	m.Y0 += y
}

// Rotate a Matrix using x and y
func (m Matrix) Rotate(angle float64) Matrix {
	return Rotate(angle).Multiply(m)
}

// Shear a Matrix using x and y
func (m Matrix) Shear(x, y float64) Matrix {
	return Shear(x, y).Multiply(m)
}
