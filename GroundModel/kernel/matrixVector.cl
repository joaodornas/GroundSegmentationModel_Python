//////////////////////////////////////
// EVAL - MATRIX VECTOR 
//////////////////////////////////////
kernel void Eval_matrixVector (
    global const float * restrict M, 
	int _rows,
    global const float * restrict V,
	int _cols,
    global float * restrict W
)

{

	int Aind = get_global_id(0);
    //int Bind = get_global_id(1);
    int Cind = Aind;

	if (Aind < _rows) 
	{ 
   
		float sum = 0;

		for (int i = 0; i < _cols; i++)
		{
			sum += M[Aind + i*_rows] * V[i];
		}

		W[Cind] = sum;

	}

}