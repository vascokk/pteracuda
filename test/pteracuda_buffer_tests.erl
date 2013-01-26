    -module(pteracuda_buffer_tests).

-compile(export_all).

-include_lib("eunit/include/eunit.hrl").

%%
%% Erlang Buffer API tests
%%

float_matrix_buffer_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(1000),
    Cols = random:uniform(1000),
    M = [[random:uniform(1000)+0.001 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, C} = pteracuda_context:new(),
    {Res, Buf_M} = pteracuda_buffer:new(matrix, float, row_major, Rows, Cols),
    ?assertEqual(Res,ok),
    ?assertEqual(ok, pteracuda_buffer:write(Buf_M, M)),
    ?assertEqual({ok, M}, pteracuda_buffer:read(Buf_M)),
    ?assertEqual(ok,  pteracuda_buffer:destroy(Buf_M)),
    ?assertEqual(ok,  pteracuda_context:destroy(C)).

float_matrix_buffer_2_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(1000),
    Cols = random:uniform(1000),
    M = [[random:uniform(1000)+0.001 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, C} = pteracuda_context:new(),
    {Res, Buf_M} = pteracuda_buffer:new(matrix, float, row_major, M), 
    ?assertEqual(Res,ok),
    ?assertEqual({ok, M}, pteracuda_buffer:read(Buf_M)),
    ?assertEqual(ok, pteracuda_buffer:destroy(Buf_M)),
    ?assertEqual(ok, pteracuda_context:destroy(C)).


int_matrix_buffer_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(1000),
    Cols = random:uniform(1000),
    M = [[random:uniform(1000) || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    {ok, C} = pteracuda_context:new(),
    {Res, Buf_M} = pteracuda_buffer:new(matrix, integer, row_major, M), 
    ?assertEqual(Res, ok),
    ?assertEqual({ok, M} , pteracuda_buffer:read(Buf_M)),
    ok = pteracuda_buffer:destroy(Buf_M),
    ok = pteracuda_context:destroy(C).

int_matrix_buffer_negative_test() ->
    {T1, T2, T3} = erlang:now(),
    random:seed(T1, T2, T3),
    Rows = random:uniform(10),
    Cols = random:uniform(10),
    M = [[random:uniform(1000)+0.001 || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)],
    %?debugMsg(io_lib:format("~n M result:~p",[M])),
    {ok, C} = pteracuda_context:new(),
    {Res, Buf_M} = pteracuda_buffer:new(matrix, integer, row_major, M), 
    ?assertEqual(Res, ok),   
    ?assertNot({ok,M} =:= pteracuda_buffer:read(Buf_M)), %returns error, we generated float but the storage is integer 
    ok = pteracuda_buffer:destroy(Buf_M),
    ok = pteracuda_context:destroy(C).

ones_1_test() ->
    {Res, Buf_ones} = pteracuda_buffer:ones(float, 5),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [1.0,1.0,1.0,1.0,1.0]} , pteracuda_buffer:read(Buf_ones)),
    ok = pteracuda_buffer:destroy(Buf_ones).

ones_2_test() ->
    {Res, Buf_ones} = pteracuda_buffer:ones(matrix, float, row_major, 2, 3),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [[1.0,1.0,1.0],[1.0,1.0,1.0]]} , pteracuda_buffer:read(Buf_ones)),
    ok = pteracuda_buffer:destroy(Buf_ones).

zeros_1_test() ->
    {Res, Buf_zeros} = pteracuda_buffer:zeros(float, 5),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [0.0,0.0,0.0,0.0,0.0]} , pteracuda_buffer:read(Buf_zeros)),
    ok = pteracuda_buffer:destroy(Buf_zeros).

zeros_2_test() ->
    {Res, Buf_zeros} = pteracuda_buffer:zeros(matrix, float, row_major, 2, 3),
    ?assertEqual(Res,ok),
    ?assertEqual({ok, [[0.0,0.0,0.0],[0.0,0.0,0.0]]} , pteracuda_buffer:read(Buf_zeros)),
    ok = pteracuda_buffer:destroy(Buf_zeros).
