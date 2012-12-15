-module(pteracuda_nifs_tests).

-compile(export_all).

-include_lib("eunit/include/eunit.hrl").
 
-define(setup(F), {setup, fun start/0, fun stop/1, F}).  

%%%%%%%%%%%%%%%%%%%%%%%
%%% SETUP FUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%
start() -> 
	{ok, Ctx} = pteracuda_nifs:new_context(),
	Ctx.
 
stop(Ctx) -> 
    ok = pteracuda_nifs:destroy_context(Ctx),
    ok.


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
    %{ok, 5} = pteracuda_nifs:buffer_size(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).


create_write_destroy_matrix_float_test() ->
    {ok, Buf} = pteracuda_nifs:new_matrix_int_buffer(4,4),
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    pteracuda_nifs:write_buffer(Buf, A),
    %{ok, 5} = pteracuda_nifs:buffer_size(Buf),
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
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    {ok, Buf} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, A} = pteracuda_nifs:read_buffer(Buf),
    ok = pteracuda_nifs:destroy_buffer(Buf).


mmul_test()->
    {ok, Ctx} = pteracuda_nifs:new_context(),
    A = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    B = [[16.0,2.0,3.0,13.0],[5.0,11.0,10.0,8.0],[9.0,7.0,6.0,12.0],[4.0,14.0,15.0,1.0]],
    _m = 4,%num_rows_A
    _k = 4,%num_cols_A
    _n = 4,%num_cols_B
    %expected result C:
    C= [[345.0, 257.0, 281.0, 273.0],[257.0, 313.0, 305.0, 281.0],[281.0, 305.0, 313.0, 257.0],[273.0, 281.0, 257.0, 345.0]],
    %this is element-wise multiplication result: C = [[256.0, 4.0, 9.0, 169.0],[25.0, 121.0, 100.0, 64.0],[81.0, 49.0, 36.0, 144.0],[16.0, 196.0, 225.0, 1.0]],
    {ok, Buf_A} = pteracuda_nifs:new_matrix_float_buffer(A),
    {ok, Buf_B} = pteracuda_nifs:new_matrix_float_buffer(B),
    {ok, Buf_C} = pteracuda_nifs:new_matrix_float_buffer(4,4),
    ok = pteracuda_nifs:mmul(Ctx, Buf_A, Buf_B, Buf_C, _m, _k, _n),
    {ok, C} = pteracuda_nifs:read_buffer(Buf_C),
    ok = pteracuda_nifs:destroy_buffer(Buf_A),
    ok = pteracuda_nifs:destroy_buffer(Buf_B),
    ok = pteracuda_nifs:destroy_buffer(Buf_C),
    pteracuda_nifs:destroy_context(Ctx).
