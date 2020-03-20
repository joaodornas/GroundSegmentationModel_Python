/////////////////////////////////////////////
// EVAL - MATRIX COPY
/////////////////////////////////////////////
kernel void Eval_matrixCopy (
    global const float * restrict A,
	global float * restrict C,
	int ldc
)
{
    
	int Aind = get_global_id(0);
    int Bind = get_global_id(1);
    int Cind = Aind + Bind*ldc;
	 
	C[Cind] = A[Cind];
	
}