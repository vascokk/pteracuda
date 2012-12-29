-module(pteracuda_nifs).

-include("include/pteracuda.hrl").
-define(NIF_API_VERSION, 2).
-define(MISSING_NIF, throw({error, missing_nif})).

-include_lib("eunit/include/eunit.hrl").


-on_load(init/0).

-export([init/0]).

%% API
-export([new_context/0,
         new_context/1,
         destroy_context/1]).

-export([new_int_buffer/0, 
         new_int_buffer/1,
         new_string_buffer/0,
         new_float_buffer/0,
         new_float_buffer/1,
         destroy_buffer/1,
         buffer_size/1]).

-export([write_buffer/2,
         buffer_delete/2,
         buffer_insert/3,
         read_buffer/1,
         clear_buffer/1,
         copy_buffer/2]).

-export([sort_buffer/2,
         buffer_contains/3,
         buffer_intersection/3,
         buffer_minmax/2]).

-export([new_matrix_int_buffer/2,
         new_matrix_float_buffer/2,
         new_matrix_int_buffer/3,
         new_matrix_float_buffer/3]).

-export([gemm/11, 
         gemv/9, 
         saxpy/4, 
         transpose/3, 
         geam/10,
         smm/4]).

-export([sigmoid/3,
         tanh/3]).

-type transpose_op() :: ?TRANSPOSE |?NO_TRANSPOSE | ?CONJUGATE_TRANSPOSE.
-type orientation_C() :: ?ROW_MAJOR | ?COLUMN_MAJOR.

new_context() ->
    ?MISSING_NIF.

new_context(_DeviceNum) ->
    ?MISSING_NIF.

destroy_context(_Ctx) ->
    ?MISSING_NIF.

new_int_buffer() ->
    ?MISSING_NIF.


new_int_buffer(_size) ->
    ?MISSING_NIF.

new_string_buffer() ->
    ?MISSING_NIF.

new_float_buffer() ->
    ?MISSING_NIF.

new_float_buffer(_size) ->
    ?MISSING_NIF.

destroy_buffer(_Buffer) ->
    ?MISSING_NIF.

buffer_size(_Buffer) ->
    ?MISSING_NIF.

-spec read_buffer(term()) -> {ok, int_vector() | float_vector() | int_matrix() | float_matrix()}.
read_buffer(_Buffer) ->
    ?MISSING_NIF.

-spec write_buffer(term(), int_vector() | float_vector() | int_matrix() | float_matrix()) -> ok.
write_buffer(_Buffer, _Data) ->
    ?MISSING_NIF.

buffer_delete(_Buffer, _Pos) ->
    ?MISSING_NIF.

buffer_insert(_Buffer, _Pos, _Value) ->
    ?MISSING_NIF.

sort_buffer(_Ctx, _Buffer) ->
    ?MISSING_NIF.

clear_buffer(_Buffer) ->
    ?MISSING_NIF.

copy_buffer(_From, _To) ->
    ?MISSING_NIF.

buffer_contains(_Ctx, _Buffer, _Value) ->
    ?MISSING_NIF.

buffer_intersection(_Ctx, _First, _Second) ->
    ?MISSING_NIF.

buffer_minmax(_Ctx, _Buffer) ->
    ?MISSING_NIF.

%% Matrices
-spec new_matrix_int_buffer(int_matrix(), orientation_C()) -> {ok, term()}.
new_matrix_int_buffer(_A, _orientation) ->
    ?MISSING_NIF.

-spec new_matrix_float_buffer(float_matrix(), orientation_C()) -> {ok, term()}.
new_matrix_float_buffer(_A, _orientation) ->
    ?MISSING_NIF.   

-spec new_matrix_int_buffer(matrix_rows(), matrix_columns(), orientation_C()) -> {ok, term()}.
new_matrix_int_buffer(_m, _n, _orientation) ->
    ?MISSING_NIF.

-spec new_matrix_float_buffer(matrix_rows(), matrix_columns(), orientation_C()) -> {ok, term()}.
new_matrix_float_buffer(_m, _n, _orientation) ->
    ?MISSING_NIF.    

-spec gemm(term(), transpose_op(), transpose_op(), matrix_rows(), matrix_columns(), matrix_rows(), float(), float_matrix(), float_matrix(), float(), float_matrix()) -> ok.
gemm(_Ctx, _transpose_op_A, _transpose_op_B, _m, _n, _k, _alpha, _A, _B, _beta, _C ) ->
    ?MISSING_NIF.

-spec gemv(term(), transpose_op(), matrix_rows(), matrix_columns(), float(), float_matrix(), float_vector(), float(), float_vector()) -> ok.
gemv(_Ctx, _transpose, _m, _n, _alpha, _A, _X, _beta, _Y) ->
    ?MISSING_NIF.

-spec saxpy(term(), float(), float_vector(), float_vector()) -> ok.
saxpy(_Ctx, _a, _X, _Y) ->
    ?MISSING_NIF.

-spec transpose(term(), float_matrix(), float_matrix()) -> ok.
transpose(_Ctx, _A, _B) ->
    ?MISSING_NIF.    

-spec geam(term(), transpose_op(), transpose_op(), matrix_rows(), matrix_columns(), float(), float_matrix(), float(), float_matrix(),  float_matrix()) -> ok.
geam(_Ctx, _transpose_op_A, _transpose_op_B, _m, _n, _alpha, _A, _beta, _B, _C ) ->
    ?MISSING_NIF.

-spec smm(term(), float(), float_matrix(), float_matrix()) -> ok.
smm(_Ctx, _alpha, _A, _B) ->
    ?MISSING_NIF.    


-spec sigmoid(term(), float_vector()|float_matrix(), float_vector()|float_matrix()) -> ok.
sigmoid(_Ctx, _A, _B) ->
    ?MISSING_NIF.

-spec tanh(term(), float_vector()|float_matrix(), float_vector()|float_matrix()) -> ok.
tanh(_Ctx, _A, _B) ->
    ?MISSING_NIF.


init() ->
    PrivDir = case code:priv_dir(pteracuda) of
                  {error, bad_name} ->
                      D = filename:dirname(code:which(?MODULE)),
                      filename:join([D, "..", "priv"]);
                  Dir ->
                      Dir
              end,
    SoName = filename:join([PrivDir, "pteracuda_nifs"]),
    erlang:load_nif(SoName, ?NIF_API_VERSION).

