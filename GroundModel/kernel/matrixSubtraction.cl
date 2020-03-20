/////////////////////////////////////////
// EVAL - MATRIX SUBTRACTION 
/////////////////////////////////////////
kernel void Eval_matrixSubtraction (
	global const float * restrict A,
    global const float * restrict B, 
    global float * restrict C,
    int ldc
)
{
  
    int Aind = get_global_id(0);
    int Bind = get_global_id(1);
    int Cind = Aind + Bind*ldc;

	C[Cind] = A[Cind] - B[Cind]; 

}