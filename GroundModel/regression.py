
import theano
import theano.tensor as tensor
from theano import function, config, shared, tensor
import numpy as np
import csv
import pyopencl as cl

#/////////////////////////////////////////////////////////
#// GAUSSIAN REGRESSION MODEL ////////////////////////////
#/////////////////////////////////////////////////////////

def train():

    import globals

    def saveMatrix(M,saveArrayInformation,which):

         if saveArrayInformation:
            np.savetxt('sanityCheck/' + which + '.txt', M, delimiter=',')

    getDim = function([],globals.Sp.OP.shape[0])
    dim_X = np.uint16(getDim())

    getDim = function([],globals.Test.OP.shape[0])
    dim_Xstar = np.uint16(getDim())

    getX = function([],globals.Sp.OP)
    X = np.array(getX())

    getXstar = function([],globals.Test.OP)
    Xstar = np.array(getXstar())

    getZ = function([],globals.Sp.Z)
    Z = np.array(getZ())

    saveMatrix(X,globals.saveArrayInformation,'X')
    saveMatrix(Xstar,globals.saveArrayInformation,'Xstar')
    saveMatrix(Z,globals.saveArrayInformation,'Z')

    covXX = np.zeros((dim_X,dim_X), dtype=config.floatX)
    covXsX = np.zeros((dim_Xstar,dim_X), dtype=config.floatX)
    covXXs = np.zeros((dim_X,dim_Xstar), dtype=config.floatX)
    covXsXs = np.zeros((dim_Xstar,dim_Xstar), dtype=config.floatX)

    Fstar = np.zeros((dim_Xstar), dtype=config.floatX)
    Vstar = np.zeros((dim_Xstar,dim_Xstar), dtype=config.floatX)

    platforms = cl.get_platforms()
    ctx = cl.Context(dev_type=cl.device_type.GPU,properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    X_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    Xstar_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Xstar)
    Z_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Z)

    covXX_d = cl.Buffer(ctx, mf.READ_WRITE, covXX.nbytes)
    covXsX_d = cl.Buffer(ctx, mf.READ_WRITE, covXsX.nbytes)
    covXXs_d = cl.Buffer(ctx, mf.READ_WRITE, covXXs.nbytes)
    covXsXs_d = cl.Buffer(ctx, mf.READ_WRITE, covXsXs.nbytes)
    
    Fstar_d = cl.Buffer(ctx, mf.READ_WRITE, Fstar.nbytes)
    Vstar_d = cl.Buffer(ctx, mf.READ_WRITE, Vstar.nbytes)

    sqrExpKernelNoise = """

    kernel void Gauss_SquareExponentialKernelNoise (
    global const float * restrict X,
	ushort _dim_size_X,
	global const float * restrict star,
	ushort _dim_size_star,
	global float * restrict K_X_star,
	float sigma_f,
	float sigma_n,
	float length_scale
)

{ 
	int Aind = get_global_id(0);
    int Bind = get_global_id(1);

	int Cind = Aind + Bind*_dim_size_X;

	if ( (Aind < _dim_size_X) && (Bind < _dim_size_star) )
	{ 

		float d = (X[Aind] - star[Bind]) * (X[Aind] - star[Bind]);

		float k = (sigma_f) * exp( (-1 / (2 * length_scale * length_scale)) * d);

		if (Aind == Bind)
		{ 
			k = k + (sigma_n);
		}
		
		K_X_star[Cind] = k;


	}


}

    """

    sqrExpKernel = """
    
    kernel void Gauss_SquareExponentialKernel (
    global const float * restrict X,
	ushort _dim_size_X,
	global const float * restrict star,
	ushort _dim_size_star,
	global float * restrict K_X_star,
	float sigma_f,
	float length_scale
)

{ 
	int Aind = get_global_id(0);
    int Bind = get_global_id(1);

	int Cind = Aind*_dim_size_star + Bind;

	if ( (Aind < _dim_size_X) && (Bind < _dim_size_star) )
	{ 

		float d = (X[Aind] - star[Bind]) * (X[Aind] - star[Bind]);

		float k = (sigma_f) * exp( (-1 / (2 * length_scale * length_scale)) * d);

		K_X_star[Cind] = k;

	}


}

"""

    matrixCopy = """

    kernel void Eval_matrixCopy (
    global const float * restrict A,
	global float * restrict C,
	ushort ldc
    )
    {
    
	    int Aind = get_global_id(0);
        int Bind = get_global_id(1);
        int Cind = Aind + Bind*ldc;
	 
	    C[Cind] = A[Cind];
	
    }

    """

    matrixMultiply = """

    kernel void GEMM_matrixMultiply (
        global const float * restrict A,
        ushort _rows_A,   
        global const float * restrict B,
        ushort _cols_B,  
        global float * restrict C,
        ushort _dim_K     
    )
    {
    
        int Aind = get_global_id(0);
        int Bind = get_global_id(1);
        int Cind = Aind*_cols_B + Bind;
        
	    if ( (Aind < _rows_A) && (Bind < _cols_B) )
        { 
		    float sum = 0.0f;

		    for (int i = 0; i < _dim_K; i++)
		    {
                float a = A[Aind*_dim_K + i];

                float b = B[i*_cols_B + Bind];

			    sum += a * b; 

		    }

		    C[Cind] = sum;

	    }

    }
    
    """

    matrixVector = """

    kernel void Eval_matrixVector (
        global const float * restrict M, 
	    ushort _rows,
        global const float * restrict V,
	    ushort _cols,
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
			    sum += M[Aind*_cols + i] * V[i];
		    }

		    W[Cind] = sum;

	    }

    }

    """
    matrixSubtraction = """

    kernel void Eval_matrixSubtraction (
	    global const float * restrict A,
        global const float * restrict B, 
        global float * restrict C,
        ushort ldc
    )
    {
  
        int Aind = get_global_id(0);
        int Bind = get_global_id(1);
        int Cind = Aind + Bind*ldc;

	    C[Cind] = A[Cind] - B[Cind]; 

    }

    """

    prg_sqrExpKernelNoise = cl.Program(ctx,sqrExpKernelNoise).build()
    prg_sqrExpKernel = cl.Program(ctx,sqrExpKernel).build()
    # invert matrix
    prg_matrixCopy = cl.Program(ctx,matrixCopy).build()
    prg_matrixMultiply = cl.Program(ctx,matrixMultiply).build()
    prg_matrixVector = cl.Program(ctx,matrixVector).build()
    prg_matrixSubtraction = cl.Program(ctx,matrixSubtraction).build()
   
    ###########################
    ### COV XX
    ###########################

    prg_sqrExpKernelNoise.Gauss_SquareExponentialKernelNoise.set_scalar_arg_dtypes( [None, None, None, None, None, np.float32, np.float32, np.float32] )
    prg_sqrExpKernelNoise.Gauss_SquareExponentialKernelNoise(queue, covXX.shape, None, X_d, dim_X, X_d, dim_X, covXX_d, globals.sigma_f, globals.sigma_n, globals.length_scale)

    cl.enqueue_copy(queue, covXX, covXX_d)

    queue.finish()

    saveMatrix(covXX,globals.saveArrayInformation,'covXX')

    ###########################
    ### COV XsXs
    ###########################

    prg_sqrExpKernel.Gauss_SquareExponentialKernel.set_scalar_arg_dtypes( [None, np.int16, None, np.int16, None, np.float32, np.float32] )

    prg_sqrExpKernel.Gauss_SquareExponentialKernel(queue, covXsXs.shape, None, Xstar_d, dim_Xstar, Xstar_d, dim_Xstar, covXsXs_d, globals.sigma_f, globals.length_scale)

    cl.enqueue_copy(queue, covXsXs, covXsXs_d)

    queue.finish()

    saveMatrix(covXsXs,globals.saveArrayInformation,'covXsXs')

    ###########################
    ### COV XXs
    ###########################

    prg_sqrExpKernel.Gauss_SquareExponentialKernel.set_scalar_arg_dtypes( [None, np.int16, None, np.int16, None, np.float32, np.float32] )

    prg_sqrExpKernel.Gauss_SquareExponentialKernel(queue, covXXs.shape, None, X_d, dim_X, Xstar_d, dim_Xstar, covXXs_d, globals.sigma_f, globals.length_scale)

    cl.enqueue_copy(queue, covXXs, covXXs_d)

    queue.finish()

    saveMatrix(covXXs,globals.saveArrayInformation,'covXXs')

    ###########################
    ### COV XsX
    ###########################

    prg_sqrExpKernel.Gauss_SquareExponentialKernel.set_scalar_arg_dtypes( [None, np.int16, None, np.int16, None, np.float32, np.float32] )

    prg_sqrExpKernel.Gauss_SquareExponentialKernel(queue, covXsX.shape, None, Xstar_d, dim_Xstar, X_d, dim_X, covXsX_d, globals.sigma_f, globals.length_scale)

    cl.enqueue_copy(queue, covXsX, covXsX_d)

    queue.finish()

    saveMatrix(covXsX,globals.saveArrayInformation,'covXsX')

    #########################################################################################################################################################################
    ### COV XX Inverse
    #########################################################################################################################################################################

    A = theano.tensor.dmatrix('A')
    invA = theano.tensor.nlinalg.matrix_inverse(A)
    invFunction = theano.function([A], invA)

    covXXinv = invFunction(covXX)

    saveMatrix(covXXinv,globals.saveArrayInformation,'covXXinv')

    ###########################
    ### COV XsX * XX Inverse
    ###########################

    covXXinv_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(covXXinv).astype(np.float32))
    covXsX_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(covXsX).astype(np.float32))
    covInvProduct = np.zeros((dim_Xstar,dim_X), dtype=config.floatX)
    covInvProduct_d = cl.Buffer(ctx, mf.WRITE_ONLY, covInvProduct.nbytes)

    prg_matrixMultiply.GEMM_matrixMultiply.set_scalar_arg_dtypes( [None, np.int16, None, np.int16, None, np.int16] )

    prg_matrixMultiply.GEMM_matrixMultiply(queue, covInvProduct.shape, None, covXsX_d, dim_Xstar, covXXinv_d, dim_X, covInvProduct_d, dim_X)

    cl.enqueue_copy(queue, covInvProduct, covInvProduct_d)

    queue.finish()

    saveMatrix(covInvProduct,globals.saveArrayInformation,'covInvProduct')

    ####################################
    ### COV ( XsX * XX Inverse ) * XXs
    ####################################

    covInvProduct_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(covInvProduct).astype(np.float32))
    covXXs_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(covXXs).astype(np.float32))
    covProduct = np.zeros((dim_Xstar,dim_Xstar), dtype=config.floatX)
    covProduct_d = cl.Buffer(ctx, mf.WRITE_ONLY, covProduct.nbytes)

    prg_matrixMultiply.GEMM_matrixMultiply.set_scalar_arg_dtypes( [None, np.int16, None, np.int16, None, np.int16] )

    prg_matrixMultiply.GEMM_matrixMultiply(queue, covProduct.shape, None, covInvProduct_d, dim_Xstar, covXXs_d, dim_Xstar, covProduct_d,dim_X)

    cl.enqueue_copy(queue, covProduct, covProduct_d)

    queue.finish()

    saveMatrix(covProduct,globals.saveArrayInformation,'covProduct')

    ###############################################
    ### COV XsXs - ( ( XsX * XX Inverse ) * XXs )
    ###############################################

    prg_matrixSubtraction.Eval_matrixSubtraction.set_scalar_arg_dtypes( [None, None, None, np.int16] )

    prg_matrixSubtraction.Eval_matrixSubtraction(queue, Vstar.shape, None, covXsXs_d, covProduct_d, Vstar_d, dim_Xstar)

    cl.enqueue_copy(queue, Vstar, Vstar_d)

    queue.finish()
    
    saveMatrix(Vstar,globals.saveArrayInformation,'Vstar')

    ####################################
    ### COV ( XsX * XX Inverse ) * Z
    ####################################

    prg_matrixVector.Eval_matrixVector.set_scalar_arg_dtypes( [None, np.int16, None, np.int16, None] )

    prg_matrixVector.Eval_matrixVector(queue, Fstar.shape, None, covInvProduct_d, dim_Xstar, Z_d, dim_X, Fstar_d)

    cl.enqueue_copy(queue, Fstar, Fstar_d)

    queue.finish()

    saveMatrix(Fstar,globals.saveArrayInformation,'Fstar')

    ####################################
    ### EVAL
    ####################################

    inliers = np.array(np.arange(dim_Xstar),dtype=bool)

    getTest_Z = function([],globals.Test.Z)
    Test_Z = getTest_Z()

    count_T = 0
    count_F = 0

    for i in range(0, dim_Xstar):

        if ( (Vstar[i,i] < globals.TModel) & ( ( (Test_Z[i] - Fstar[i]) / np.sqrt(globals.sigma_n + Vstar[i,i]) ) < globals.TData ) ):

            inliers[i] = True

            count_T = count_T + 1

        else:

            inliers[i] = False

            count_F = count_F + 1

    globals.Snew.X = globals.Test.X[inliers]
    globals.Snew.Y = globals.Test.Y[inliers]
    globals.Snew.Z = globals.Test.Z[inliers]
    globals.Snew.OP = globals.Test.OP[inliers]
    globals.Snew.idx = globals.Test.idx[inliers]




