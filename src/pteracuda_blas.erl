-module(pteracuda_blas).

-include("pteracuda_internals.hrl").

-export([gemm/11, gemv/9, saxpy/4]).

-type transpose_op() :: transpose | no_transpose | conjugate_transpose.


gemm(#pc_context{ref=Ctx}, Transpose_A, Transpose_B, _m, _n, _k, _alpha, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B}, _beta, #pc_buffer{ref=Buf_C}) ->
	case Transpose_A of 
		transpose -> _transp_A = ?TRANSPOSE;
		no_transpose -> _transp_A = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_A = ?CONJUGATE_TRANSPOSE
	end,
	case Transpose_B of 
		transpose -> _transp_B = ?TRANSPOSE;
		no_transpose -> _transp_B = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_B = ?CONJUGATE_TRANSPOSE
	end,
	pteracuda_nifs:gemm(Ctx, _transp_A, _transp_B, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C).

gemv(#pc_context{ref=Ctx}, Transpose_A , _m, _n, _alpha, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_X}, _beta, #pc_buffer{ref=Buf_Y}) ->
	case Transpose_A of 
		transpose -> _transp_A = ?TRANSPOSE;
		no_transpose -> _transp_A = ?NO_TRANSPOSE;
		conjugate_transpose -> _transp_A = ?CONJUGATE_TRANSPOSE
	end,
	pteracuda_nifs:gemv(Ctx, _transp_A , _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y).

saxpy(#pc_context{ref=Ctx}, _a, #pc_buffer{ref=Buf_X}, #pc_buffer{ref=Buf_Y}) ->
	pteracuda_nifs:saxpy(Ctx, _a, Buf_X, Buf_Y).