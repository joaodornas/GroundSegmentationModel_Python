//////////////////////////////////////
// EVAL - MATRIX MULTIPLY 
//////////////////////////////////////
kernel void GEMM_matrixMultiply (
    global const float * restrict A,
    int _rows_A,   
    global const float * restrict B,
    int _cols_B,  
    global float * restrict C,
    int _dim_K     
)
{
    
    int Aind = get_global_id(0);
    int Bind = get_global_id(1);
    int Cind = Aind + Bind*_rows_A;

	if ( (Aind < _rows_A) && (Bind < _cols_B) )
    { 
		float sum = 0;

		for (int i = 0; i < _dim_K; i++)
		{
			sum += A[i*_rows_A + Aind] * B[i + Bind*_dim_K];
		}

		C[Cind] = sum;

	}

}