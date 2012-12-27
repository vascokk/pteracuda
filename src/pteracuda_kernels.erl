-module(pteracuda_kernels).

-include("include/pteracuda.hrl").

-export([sigmoid/3,
		 tanh/3]).

-spec sigmoid(context(), float_vector_buffer()|float_matrix_buffer(), float_vector_buffer()|float_matrix_buffer()) -> ok. 
sigmoid(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B})->
	%{ok, Ctx2} = pteracuda_nifs:new_context(),
	pteracuda_nifs:sigmoid(Ctx, Buf_A, Buf_B).   	

-spec tanh(context(), float_vector_buffer()|float_matrix_buffer(), float_vector_buffer()|float_matrix_buffer()) -> ok.
tanh(#pc_context{ref=Ctx}, #pc_buffer{ref=Buf_A}, #pc_buffer{ref=Buf_B})->
	pteracuda_nifs:tanh(Ctx, Buf_A, Buf_B).
