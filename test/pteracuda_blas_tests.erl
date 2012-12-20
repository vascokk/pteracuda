-module(pteracuda_blas_tests).

-compile(export_all).

-include("src/pteracuda_internals.hrl").

-include_lib("eunit/include/eunit.hrl").

-define(setup(F), {setup, fun start/0, fun stop/1, F}).


transpose([[]|_]) -> [];
transpose(M) ->
  [lists:map(fun hd/1, M) | transpose(lists:map(fun tl/1, M))].


func(RowM1,M2)->
  F2 = fun(_RowM1,_RowM2) -> [X*Y || {X,Y}<-lists:zip(_RowM1,_RowM2)] end,
  F1 = fun(_RowM1,_M2) -> [lists:sum(F2(_RowM1,_RowM2)) || _RowM2<-_M2 ] end,
	F1(RowM1,M2).
	
mmul_cpu([],Acc,MR)->
		lists:reverse(Acc);
mmul_cpu([H|T],Acc, MR)->		
		L = func(H,MR),
		mmul_cpu(T,[L|Acc],MR).


benchmark_test_() ->
          {timeout, 60*60,
           fun() ->
                  mmul()
           end}.	

mmul() ->
   	_m = 300,
    _k = 300,
    _n = 300,
    _alpha = 1.0,
    _beta= 0.0,
	  {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = _m,
    Cols = _k,
    M1 = [[random:uniform(1000) || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    M2 = [[random:uniform(1000) || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],

    %% CPU test
    Fun = fun(M1,M2)-> mmul_cpu(M1,[],transpose(M2)) end,
    {Time1, _} = timer:tc(Fun,[M1,M2]),
    %?debugMsg(io_lib:format("~n M result:~p",[ResM])),
    ?debugMsg(io_lib:format("~n Execution time Erlang(CPU):~p",[Time1])),

    %% GPU CUBLAS test
    {ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, Buf_M1} = pteracuda_nifs:new_matrix_float_buffer(M1),
    {ok, Buf_M2} =  pteracuda_nifs:new_matrix_float_buffer(M2),
    {ok, Buf_C} = pteracuda_nifs:new_matrix_float_buffer(_m,_n),

    %ok = pteracuda_nifs:gemm(Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {Time2, _} = timer:tc(pteracuda_nifs, gemm, [Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_M1, Buf_M2, _beta, Buf_C]),
    ?debugMsg(io_lib:format("~n Execution time CUDA(GPU):~p",[Time2])),
    
    ok = pteracuda_nifs:destroy_buffer(Buf_M1),
    ok = pteracuda_nifs:destroy_buffer(Buf_M2),
    ok = pteracuda_nifs:destroy_buffer(Buf_C),
    pteracuda_nifs:destroy_context(Ctx).