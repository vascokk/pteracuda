-module(pteracuda_nifs_tests).

-compile(export_all).

-include("pteracuda.hrl").
-include_lib("eunit/include/eunit.hrl").


create_destroy_test() ->
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_destroy_float_test() ->	
    {ok, Buf} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:destroy_buffer(Buf).


create_write_destroy_test() ->
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),
    pteracuda_nifs:write_buffer(Buf, [1,2,3,4,5]),
    {ok, 5} = pteracuda_nifs:buffer_size(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_write_destroy_float_test() ->
    {ok, Buf} = pteracuda_nifs:new_float_buffer(),
    pteracuda_nifs:write_buffer(Buf, [0.01, 0.002, 0.0003, 0.4, 1.5]),
    {ok, 5} = pteracuda_nifs:buffer_size(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_write_delete_test() ->
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf, [1,2,3,4,5]),
    ok = pteracuda_nifs:buffer_delete(Buf, 1),
    {ok, [1,3,4,5]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:buffer_delete(Buf, 0),
    {ok, [3,4,5]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_write_delete_float_test() ->
    {ok, Buf} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf, [1.1,1.2,1.3,1.4,1.5]),
    ok = pteracuda_nifs:buffer_delete(Buf, 1),
    {ok, [1.1,1.3,1.4,1.5]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:buffer_delete(Buf, 0),
    {ok, [1.3,1.4,1.5]} = pteracuda_nifs:read_buffer(Buf),
    pteracuda_nifs:destroy_buffer(Buf).

insert_test() ->
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),
    ok = pteracuda_nifs:buffer_insert(Buf, 0, 1),
    error = pteracuda_nifs:buffer_insert(Buf, 5, 2),
    {ok, [1]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:clear_buffer(Buf),
    ok = pteracuda_nifs:write_buffer(Buf, [1,2,3,4,5]),
    ok = pteracuda_nifs:buffer_insert(Buf, 2, 6),
    {ok, [1,2,6,3,4,5]} = pteracuda_nifs:read_buffer(Buf),
    pteracuda_nifs:destroy_buffer(Buf).

insert_float_test() ->
    {ok, Buf} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:buffer_insert(Buf, 0, 1.0),
    error = pteracuda_nifs:buffer_insert(Buf, 5, 2.0),
    {ok, [1.0]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:clear_buffer(Buf),
    ok = pteracuda_nifs:write_buffer(Buf, [1.0,2.0,3.0,4.0,5.0]),
    ok = pteracuda_nifs:buffer_insert(Buf, 2, 6.0),
    {ok, [1.0,2.0,6.0,3.0,4.0,5.0]} = pteracuda_nifs:read_buffer(Buf),
    pteracuda_nifs:destroy_buffer(Buf).

create_write_sort_destroy_test() ->
	{ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),    
    ok = pteracuda_nifs:write_buffer(Buf, [3,2,1,4,5]),
    {ok, 5} = pteracuda_nifs:buffer_size(Buf),
    ok = pteracuda_nifs:sort_buffer(Ctx, Buf),
    {ok, [1,2,3,4,5]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf),
    ok = pteracuda_nifs:destroy_context(Ctx).

create_write_sort_destroy_float_test() ->
	{ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, Buf} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf, [3.1,2.1,1.1,4.1,5.1]),
    {ok, 5} = pteracuda_nifs:buffer_size(Buf),
    ok = pteracuda_nifs:sort_buffer(Ctx, Buf),
    {ok, [1.1,2.1,3.1,4.1,5.1]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf),
    ok = pteracuda_nifs:destroy_context(Ctx).

create_write_clear_test() ->
	{ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf, [3,2,1,4,5]),
    {ok, 5} = pteracuda_nifs:buffer_size(Buf),
    pteracuda_nifs:clear_buffer(Buf),
    {ok, 0} = pteracuda_nifs:buffer_size(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf),
    ok = pteracuda_nifs:destroy_context(Ctx).

create_write_contains_test() ->
	{ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),    
    N = lists:seq(1, 1000),
    ok = pteracuda_nifs:write_buffer(Buf, N),
    true = pteracuda_nifs:buffer_contains(Ctx, Buf, 513),
    false = pteracuda_nifs:buffer_contains(Ctx, Buf, 1500),
    ok = pteracuda_nifs:destroy_buffer(Buf),
	ok = pteracuda_nifs:destroy_context(Ctx).

create_write_contains_float_test() ->
	{ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, Buf} = pteracuda_nifs:new_float_buffer(),
    
    N = [X + 0.0001 || X <- lists:seq(1, 1000)],
    ok = pteracuda_nifs:write_buffer(Buf, N),
    true = pteracuda_nifs:buffer_contains(Ctx, Buf, 20.0001),
    false = pteracuda_nifs:buffer_contains(Ctx, Buf, 1500.0),
    ok = pteracuda_nifs:destroy_buffer(Buf),
    ok = pteracuda_nifs:destroy_context(Ctx).

create_copy_test() ->
    {ok, Buf} = pteracuda_nifs:new_int_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf, lists:seq(1, 1000)),
    {ok, Buf1} = pteracuda_nifs:new_int_buffer(),
    ok = pteracuda_nifs:copy_buffer(Buf, Buf1),
    {ok, 1000} = pteracuda_nifs:buffer_size(Buf1),
    ok = pteracuda_nifs:destroy_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf1).

intersection_test() ->
	{ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, B1} = pteracuda_nifs:new_int_buffer(),
    {ok, B2} = pteracuda_nifs:new_int_buffer(),
    
    ok = pteracuda_nifs:write_buffer(B1, lists:seq(1, 100)),
    ok = pteracuda_nifs:write_buffer(B2, lists:seq(90, 190)),
    {ok, IB} = pteracuda_nifs:buffer_intersection(Ctx, B1, B2),
    11 = length(IB),
    
    pteracuda_nifs:destroy_buffer(B1),
    pteracuda_nifs:destroy_buffer(B2),
    pteracuda_nifs:destroy_context(Ctx).

minmax_test() ->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    {ok, B} = pteracuda_nifs:new_int_buffer(),
    
    F = fun(_, _) -> random:uniform(100) > 49 end,
    N = lists:sort(F, lists:seq(1, 5000)),
    pteracuda_nifs:write_buffer(B, N),
    pteracuda_nifs:sort_buffer(Ctx, B),
    {ok, {1, 5000}} = pteracuda_nifs:buffer_minmax(Ctx, B),
    pteracuda_nifs:destroy_buffer(B),
    pteracuda_nifs:destroy_context(Ctx).

create_destroy_matrix_test() ->
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(4,4),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_destroy_float_matrix_test() ->  
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(4,4),
    ok = pteracuda_nifs:destroy_buffer(Buf).


create_write_destroy_matrix_int_test() ->
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(4,4),
    A = [[16,2,3,13],[5,11,10,8],[9,7,6,12],[4,14,15,1]],
    pteracuda_nifs:write_buffer(Buf, A),
    ok = pteracuda_nifs:destroy_buffer(Buf).


create_write_destroy_matrix_float_test() ->
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(4,4),
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    pteracuda_nifs:write_buffer(Buf, A),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_write_read_destroy_matrix_int_test() ->
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(4,4),
    A = [[16,2,3,13],[5,11,10,8],[9,7,6,12],[4,14,15,1]],
    pteracuda_nifs:write_buffer(Buf, A),
    {ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_from_matrix_write_read_destroy_matrix_int_test() ->
    A = [[16,2,3,13],[5,11,10,8],[9,7,6,12],[4,14,15,1]],
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(A),
    {ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).
    
create_from_matrix_write_read_destroy_matrix_float_test() ->
    A = [[16.5,2.1029,3.00023,13.00001],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_matrix_float_2_test() ->
    A = [[7.0,4.0,3.0],[8.0,4.0,7.0],[15.0,6.0,99.0],[3.0,2.0,4.0]], 
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).


create_matrix_float_3_test() ->
    A = [[3.0,2.0,44.0,8.0],[5.0,7.0,12.0,21.0]], 
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).

create_float_matrix_with_int_values_test() ->
    A = [[7,4,3],[8,4,7],[15,6,99],[3,2,4]], 
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, [[7.0,4.0,3.0],[8.0,4.0,7.0],[15.0,6.0,99.0],[3.0,2.0,4.0]]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).


create_float_matrix_with_int_values_2_test() ->
    A = [[3,2,44,8],[5,7,12,21]], 
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, [[3.0,2.0,44.0,8.0],[5.0,7.0,12.0,21.0]]} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).


negative_create_float_matrix_with_wrong_dimensions_test() ->
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(4,3), %must be (4,4)
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    {error,_} = pteracuda_nifs:write_buffer(Buf, A),
    %{ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).


negative_create_float_matrix_with_wrong_dimensions_less_data_test() ->
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(4,4), 
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0]], %one row less
    {error,_} = pteracuda_nifs:write_buffer(Buf, A),
    %{ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).

%  
%Float matrix operations only supported
%
% GEMM: C = α op ( A ) op ( B ) + β C
gemm_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[3,5],[2,7],[44,12],[8,21]], %row major
    _m = 3,%num_rows_A
    _k = 4,%num_cols_A
    _n = 2,%num_cols_B
    _alpha = 1.0,
    _beta= 0.0,
    C = [[721.0, 334.0],[300.0,162.0],[4411.0,1336.0]], %row major
    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_B} = pteracuda_nifs:new_matrix_float_buffer(B),
    {ok, Buf_C} = pteracuda_nifs:new_matrix_float_buffer(_m,_n),
    ok = pteracuda_nifs:gemm(Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, C} = pteracuda_nifs:read_buffer(Buf_C),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_B),
    ok = pteracuda_nifs:destroy_buffer(Buf_C),
    pteracuda_nifs:destroy_context(Ctx).

negative_gemm_wrong_A_dim_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[3,5],[2,7],[44,12],[8,21]], %row major
    _m = 4,%num_rows_A  WRONG!!! must be 3
    _k = 4,%num_cols_A
    _n = 2,%num_cols_B
    _alpha = 1.0,
    _beta= 0.0,
    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_B} = pteracuda_nifs:new_matrix_float_buffer(B),
    {ok, Buf_C} = pteracuda_nifs:new_matrix_float_buffer(_m,_n),
    {error,_} = pteracuda_nifs:gemm(Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _k, _alpha, Buf_A, Buf_B, _beta, Buf_C),
    {ok, _} = pteracuda_nifs:read_buffer(Buf_C),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_B),
    ok = pteracuda_nifs:destroy_buffer(Buf_C),
    pteracuda_nifs:destroy_context(Ctx).


%  GEMV: y <- α op ( A ) x + β y
gemv_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    Y = [0.0, 0.0], 
    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_X} = pteracuda_nifs:new_float_buffer(),
    pteracuda_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = pteracuda_nifs:new_float_buffer(),
    pteracuda_nifs:write_buffer(Buf_Y, Y),
    ok = pteracuda_nifs:gemv(Ctx, ?NO_TRANSPOSE , _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y),
    {ok, [60.0,75.0]} = pteracuda_nifs:read_buffer(Buf_Y),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_X),
    ok = pteracuda_nifs:destroy_buffer(Buf_Y),
    pteracuda_nifs:destroy_context(Ctx).

negative_gemv_wrong_A_dim_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 5, %rows A  WRONG!!! must be 2
    _n = 4, %columns A
    _alpha = 1.0,
    _beta = 0.0,
    X = [2.0,5.0,1.0,7.0],
    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_X} = pteracuda_nifs:new_float_buffer(),
    pteracuda_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = pteracuda_nifs:new_float_buffer(),
    {error, _} = pteracuda_nifs:gemv(Ctx, ?NO_TRANSPOSE, _m, _n, _alpha, Buf_A, Buf_X, _beta, Buf_Y),
    {ok, _} = pteracuda_nifs:read_buffer(Buf_Y),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_X),
    ok = pteracuda_nifs:destroy_buffer(Buf_Y),
    pteracuda_nifs:destroy_context(Ctx).
    
%SAXPY:  y <- α * x + y
saxpy_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    _a = 2.0, %!!!! this has to be float
    X = [2.0, 5.0, 1.0, 7.0],
    Y = [0.0, 0.0, 0.0, 0.0], 
    {ok, Buf_X} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf_Y, Y),
    ok = pteracuda_nifs:saxpy(Ctx, _a, Buf_X, Buf_Y),
    {ok, [4.0, 10.0, 2.0, 14.0]} = pteracuda_nifs:read_buffer(Buf_Y),
    ok = pteracuda_nifs:destroy_buffer(Buf_X),
    ok = pteracuda_nifs:destroy_buffer(Buf_Y),
    pteracuda_nifs:destroy_context(Ctx).

negative_saxpy_sizeX_lt_sizeY_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    _a = 2.0, %!!!! this has to be float
    X = [2.0, 5.0, 1.0],
    Y = [0.0, 0.0, 0.0, 0.0], 
    {ok, Buf_X} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf_X, X),
    {ok, Buf_Y} = pteracuda_nifs:new_float_buffer(),
    ok = pteracuda_nifs:write_buffer(Buf_Y, Y),
    {error, _} = pteracuda_nifs:saxpy(Ctx, _a, Buf_X, Buf_Y),
    {ok, _} = pteracuda_nifs:read_buffer(Buf_Y),
    ok = pteracuda_nifs:destroy_buffer(Buf_X),
    ok = pteracuda_nifs:destroy_buffer(Buf_Y),
    pteracuda_nifs:destroy_context(Ctx).


%%%
%%% BLAS-like functions
%%%

%Transpose: B <- transpose(A)
transpose_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    A_transposed = [[7.0,4.0,3.0],[8.0,4.0,7.0],[15.0,6.0,99.0],[3.0,2.0,4.0]],

    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_B} = pteracuda_nifs:new_matrix_float_buffer(4,3),
    ok = pteracuda_nifs:transpose(Ctx, Buf_A, Buf_B),
    {ok, B} = pteracuda_nifs:read_buffer(Buf_B),
    ?assertEqual(B, A_transposed),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_B),
    pteracuda_nifs:destroy_context(Ctx).

% GEAM:  C = α op ( A ) + β op ( B )
% (this function is CUBLAS-specific)
geam_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[7,8,15,3],[4,4,6,2],[3,7,99,4]], %row major
    B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
    _alpha = 1.0,
    _beta = 1.0,
    _m = 3,
    _n = 4,
    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_B} = pteracuda_nifs:new_matrix_float_buffer(B),
    {ok, Buf_C} = pteracuda_nifs:new_matrix_float_buffer(_m, _n),
    ok = pteracuda_nifs:geam(Ctx, ?NO_TRANSPOSE, ?NO_TRANSPOSE, _m, _n, _alpha, Buf_A, _beta, Buf_B, Buf_C),
    {ok, C} = pteracuda_nifs:read_buffer(Buf_C),
    ?assertEqual(C, [[8.0,10.0,18.0,7.0],[9.0,10.0,13.0,10.0],[12.0,17.0,110.0,16.0]]),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_B),
    pteracuda_nifs:destroy_context(Ctx).

% smm (Scalar Matrix Multiply)
% B <- α * A
smm_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[4.0,6.0,8.0,2.0],[5.0,7.0,9.0,3.0]],
    _m = 2, %rows A
    _n = 4, %columns A
    _alpha = 5.0,
    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_B} = pteracuda_nifs:new_matrix_float_buffer(_m, _n),
    ok = pteracuda_nifs:smm(Ctx, _alpha, Buf_A, Buf_B),
    {ok, B} = pteracuda_nifs:read_buffer(Buf_B),
    ?assertEqual(B, [[20.0,30.0,40.0,10.0],[25.0,35.0,45.0,15.0]]),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_B),
    pteracuda_nifs:destroy_context(Ctx).
