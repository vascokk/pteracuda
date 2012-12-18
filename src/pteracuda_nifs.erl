-module(pteracuda_nifs).

-define(NIF_API_VERSION, 1).
-define(MISSING_NIF, throw({error, missing_nif})).

-include_lib("eunit/include/eunit.hrl").

-on_load(init/0).

-export([init/0]).

%% API
-export([new_context/0,
         new_context/1,
         destroy_context/1]).

-export([new_int_buffer/0,
         new_string_buffer/0,
         new_float_buffer/0,
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

-export([new_matrix_int_buffer/1,
         new_matrix_float_buffer/1,
         new_matrix_int_buffer/2,
         new_matrix_float_buffer/2]).

-export([mmul/7, gemv/8, saxpy/4]).

new_context() ->
    ?MISSING_NIF.

new_context(_DeviceNum) ->
    ?MISSING_NIF.

destroy_context(_Ctx) ->
    ?MISSING_NIF.

new_int_buffer() ->
    ?MISSING_NIF.

new_string_buffer() ->
    ?MISSING_NIF.

new_float_buffer() ->
    ?MISSING_NIF.

destroy_buffer(_Buffer) ->
    ?MISSING_NIF.

buffer_size(_Buffer) ->
    ?MISSING_NIF.

read_buffer(_Buffer) ->
    ?MISSING_NIF.

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

new_matrix_int_buffer(_A) ->
    ?MISSING_NIF.

new_matrix_float_buffer(_A) ->
    ?MISSING_NIF.   

new_matrix_int_buffer(_m, _n) ->
    ?MISSING_NIF.

new_matrix_float_buffer(_m, _n) ->
    ?MISSING_NIF.    

mmul(_Ctx, _A, _B, _C, _m, _k, _n) ->
    ?MISSING_NIF.

gemv(_Ctx, _m, _n, _alpha, _A, _X, _beta, _Y) ->
    ?MISSING_NIF.

saxpy(_Ctx, _a, _X, _Y) ->
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

