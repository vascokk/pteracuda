-module(pteracuda_buffer_tests).

-compile(export_all).

-include_lib("eunit/include/eunit.hrl").

create_float_matrix_buffer_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(1000),
    Cols = random:uniform(1000),
    D = [[random:uniform(1000) || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, C} = pteracuda_context:new(),
    {ok, B} = pteracuda_buffer:new(matrix_float, Rows, Cols),
    ok = pteracuda_buffer:write(B, D),
    ok = pteracuda_buffer:destroy(B),
    ok = pteracuda_context:destroy(C).

create_float_matrix_buffer_2_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(1000),
    Cols = random:uniform(1000),
    D = [[random:uniform(1000) || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, C} = pteracuda_context:new(),
    {ok, B} = pteracuda_buffer:new(matrix_float, D), 
    %ok = pteracuda_buffer:write(B, D),
    ok = pteracuda_buffer:destroy(B),
    ok = pteracuda_context:destroy(C).

