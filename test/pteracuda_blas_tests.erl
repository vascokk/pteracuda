-module(pteracuda_blas_tests).

-compile(export_all).

-include("pteracuda.hrl").

-include_lib("eunit/include/eunit.hrl").

-define(setup(F), {setup, fun start/0, fun stop/1, F}).


transpose([[]|_]) -> [];
transpose(M) ->
  [lists:map(fun hd/1, M) | transpose(lists:map(fun tl/1, M))].


transpose_gpu(Ctx, Buf_A, Buf_B) ->
  ok = pteracuda_nifs:transpose(Ctx, Buf_A, Buf_B),
  {ok, B} = pteracuda_nifs:read_buffer(Buf_B),
  B.

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

%% Matrix-matrix multiplication benchmark
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
    M1 = [[random:uniform(1000)+0.1 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    M2 = [[random:uniform(1000)+0.1 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    
    %% CPU test
    Fun = fun(M1,M2)-> mmul_cpu(M1,[],transpose(M2)) end,
    {Time1, _} = timer:tc(Fun,[M1,M2]),
    %?debugMsg(io_lib:format("~n M result:~p",[ResM])),
   
    {ok, Ctx} = pteracuda_nifs:new_context(),

    %% GPU CUBLAS test
    {ok, Buf_M1} = pteracuda_nifs:new_matrix_float_buffer(M1, ?ROW_MAJOR),
    {ok, Buf_M2} =  pteracuda_nifs:new_matrix_float_buffer(M2, ?ROW_MAJOR),
    {ok, Buf_C} = pteracuda_nifs:new_matrix_float_buffer(_m,_n, ?ROW_MAJOR),
    {Time2, _} = timer:tc(pteracuda_nifs, gemm, [Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_M1, Buf_M2, _beta, Buf_C]),


    %% CPU multiplication with GPU transpose
    {ok, Buf_M2T} = pteracuda_nifs:new_matrix_float_buffer(_n, _k, ?ROW_MAJOR),
    {Time3, _} = timer:tc(Fun,[M1,transpose_gpu(Ctx, Buf_M1, Buf_M2T)]),   

    ok = pteracuda_nifs:destroy_buffer(Buf_M1),
    ok = pteracuda_nifs:destroy_buffer(Buf_M2),
    ok = pteracuda_nifs:destroy_buffer(Buf_C),
    ok = pteracuda_nifs:destroy_buffer(Buf_M2T),

    pteracuda_nifs:destroy_context(Ctx),
    
    %%Print results
    ?debugMsg(io_lib:format("Execution time Erlang(CPU):~p",[Time1])),
    ?debugMsg(io_lib:format("Execution time CUDA(GPU):~p",[Time2])),
    ?debugMsg(io_lib:format("Execution time Erlang & CUDA transpose:~p",[Time3])).
    

%%
%% Erlang BLAS API tests
%%

% GEMM: C = α op ( A ) op ( B ) + β C
gemm_test()->
    {ok, Ctx} = pteracuda_context:new(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[3,5],[2,7],[44,12],[8,21]], %row major
    _m = 3,%num_rows_A
    _k = 4,%num_cols_A
    _n = 2,%num_cols_B
    _alpha = 1.0,
    _beta= 0.0,
    C = [[721.0, 334.0],[300.0,162.0],[4411.0,1336.0]], %row major
    {ok, Buf_A} = pteracuda_buffer:new(matrix, float, row_major,A),
    {ok, Buf_B} = pteracuda_buffer:new(matrix, float, row_major,B),
    {ok, Buf_C} = pteracuda_buffer:new(matrix, float, row_major,_m,_n),
    ok = pteracuda_blas:gemm(Ctx, no_transpose, no_transpose, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, C} = pteracuda_buffer:read(Buf_C),
    ok = pteracuda_buffer:destroy(Buf_A),
    ok = pteracuda_buffer:destroy(Buf_B),
    ok = pteracuda_buffer:destroy(Buf_C),
    ok = pteracuda_context:destroy(Ctx).


%  GEMV: y <- α op ( A ) x + β y
gemv_test()->
    {ok, Ctx} = pteracuda_context:new(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    Y = [0.0, 0.0], 
    {ok, Buf_A} = pteracuda_buffer:new(matrix, float, row_major, A),
    {ok, Buf_X} = pteracuda_buffer:new(float),
    pteracuda_buffer:write(Buf_X, X),
    {ok, Buf_Y} = pteracuda_buffer:new(float),
    pteracuda_buffer:write(Buf_Y, Y),
    ok = pteracuda_blas:gemv(Ctx, no_transpose , _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y),
    {ok, [60.0,75.0]} = pteracuda_buffer:read(Buf_Y),
    ok = pteracuda_buffer:destroy(Buf_A),
    ok = pteracuda_buffer:destroy(Buf_X),
    ok = pteracuda_buffer:destroy(Buf_Y),
    ok = pteracuda_context:destroy(Ctx).

%SAXPY:  y <- a * x + y
saxpy_test()->
    {ok, Ctx} = pteracuda_context:new(),
    _a = 2.0, %!!!! this has to be float
    X = [2.0, 5.0, 1.0, 7.0],
    Y = [0.0, 0.0, 0.0, 0.0], 
    {ok, Buf_X} = pteracuda_buffer:new(float),
    ok = pteracuda_buffer:write(Buf_X, X),
    {ok, Buf_Y} = pteracuda_buffer:new(float),
    ok = pteracuda_buffer:write(Buf_Y, Y),
    ok = pteracuda_blas:saxpy(Ctx, _a, Buf_X, Buf_Y),
    {ok, [4.0, 10.0, 2.0, 14.0]} = pteracuda_buffer:read(Buf_Y),
    ok = pteracuda_buffer:destroy(Buf_X),
    ok = pteracuda_buffer:destroy(Buf_Y),
    ok = pteracuda_context:destroy(Ctx).

% GEAM:  C = α op ( A ) + β op ( B )
geam_test()->
    {ok, Ctx} = pteracuda_context:new(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
    _alpha = 1.0,
    _beta = 1.0,
    _m = 3,
    _n = 4,
    {ok, Buf_A} = pteracuda_buffer:new(matrix, float, row_major, A),
    {ok, Buf_B} = pteracuda_buffer:new(matrix, float, row_major, B),
    {ok, Buf_C} = pteracuda_buffer:new(matrix, float, row_major, _m, _n),
    ok = pteracuda_blas:geam(Ctx, no_transpose, no_transpose, _m, _n, _alpha, Buf_A, _beta, Buf_B, Buf_C),
    {ok, C} = pteracuda_buffer:read(Buf_C),
    ?assertEqual(C, [[8.0,10.0,18.0,7.0],[9.0,10.0,13.0,10.0],[12.0,17.0,110.0,16.0]]),
    ok = pteracuda_buffer:destroy(Buf_A),
    ok = pteracuda_buffer:destroy(Buf_B),
    ok = pteracuda_context:destroy(Ctx).    

% B <- α * A
smm_test()->
    {ok, Ctx} = pteracuda_context:new(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 5.0,
    {ok, Buf_A} = pteracuda_buffer:new(matrix, float, row_major, A),
    {ok, Buf_B} = pteracuda_buffer:new(matrix, float, row_major, _m, _n),
    ok = pteracuda_blas:smm(Ctx, _alpha, Buf_A, Buf_B),
    {ok, B} = pteracuda_buffer:read(Buf_B),
    ?assertEqual(B, [[20.0,30.0,40.0,10.0],[25.0,35.0,45.0,15.0]]),
    ok = pteracuda_buffer:destroy(Buf_A),
    ok = pteracuda_buffer:destroy(Buf_B),
    ok = pteracuda_context:destroy(Ctx).

transpose_benchmark_test()->
    Rows = 500,
    Cols = 500,
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    M = [[random:uniform(1000)+0.1 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, Ctx} = pteracuda_context:new(),

    {ok, Buf_M} = pteracuda_buffer:new(matrix, float, row_major, M),
    {ok, Buf_MT} =  pteracuda_buffer:new(matrix, float, row_major, Cols, Rows),
    Fun = fun(M1) -> transpose(M1) end,
    {Time1, _} = timer:tc(pteracuda_nifs, transpose, [Ctx, Buf_M, Buf_MT]),
    {Time2, _} = timer:tc(Fun, [M]),
    ok = pteracuda_buffer:destroy(Buf_M),
    ok = pteracuda_context:destroy(Ctx),

    ?debugMsg(io_lib:format("Transpose GPU:~p",[Time1])),
    ?debugMsg(io_lib:format("Transpose CPU:~p",[Time2])).


