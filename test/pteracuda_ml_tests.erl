-module(pteracuda_ml_tests).

-compile(export_all).
-include("pteracuda.hrl").
-include_lib("eunit/include/eunit.hrl").

bin_to_num(Elem) ->
    try list_to_float(Elem)
    	catch error:badarg -> list_to_integer(Elem)
    end.

readfile(FileName, EOL) ->
    {ok, Binary} = file:read_file(FileName),
    Lines = string:tokens(erlang:binary_to_list(Binary), EOL),
    [[bin_to_num(X) || X<-string:tokens(Y, ",")] || Y<-Lines].

gd_test() ->	
	{ok, Ctx} = pteracuda_context:new(),
	TrainSet = readfile(ex2data1.txt, "\r\n"),
	_m = length(TrainSet), % number of training examples
	Theta = [0,0,0], %
	X = [ lists:append([1.0], lists:sublist(TrEx, 2)) || TrEx<-TrainSet], %[1.0] - adding the bias term
	Y = lists:append( [ lists:sublist(TrEx, 3,3) || TrEx<-TrainSet] ),  
	{ok, X_buf} = pteracuda_buffer:new(matrix, float, row_major, X),
	
	{ok, Y_buf} = pteracuda_buffer:new(matrix, float, row_major, [Y]), 
    {ok, T_buf} = pteracuda_buffer:new(matrix, float, row_major, [Theta]),
	
	ok = pteracuda_ml:gd(Ctx, T_buf, X_buf, Y_buf, length(hd(X)), length(X)),	
	{ok, Grad} = pteracuda_buffer:read(T_buf),
	pteracuda_buffer:destroy(T_buf),
	pteracuda_buffer:destroy(Y_buf),
	pteracuda_buffer:destroy(X_buf),
	pteracuda_context:destroy(Ctx),	
	?debugMsg(io_lib:format("~n Gradient:~p",[Grad])).

gd_learn_test() ->	
	{ok, Ctx} = pteracuda_context:new(),
	TrainSet = readfile(ex2data1.txt, "\r\n"),
	_m = length(TrainSet), % number of training examples
	Theta = [0,0,0], %
	X = [ lists:append([1.0], lists:sublist(TrEx, 2)) || TrEx<-TrainSet], %[1.0] - adding the bias term
	Y = lists:append( [ lists:sublist(TrEx, 3,3) || TrEx<-TrainSet] ),  
	{ok, X_buf} = pteracuda_buffer:new(matrix, float, row_major, X),
	
	{ok, Y_buf} = pteracuda_buffer:new(matrix, float, row_major, [Y]), 
    {ok, T_buf} = pteracuda_buffer:new(matrix, float, row_major, [Theta]),
	
	ok = pteracuda_ml:gd_learn(Ctx, T_buf, X_buf, Y_buf, length(hd(X)), length(X), 0.001, 401),	
	{ok, Grad} = pteracuda_buffer:read(T_buf),
	pteracuda_buffer:destroy(T_buf),
	pteracuda_buffer:destroy(Y_buf),
	pteracuda_buffer:destroy(X_buf),
	pteracuda_context:destroy(Ctx),	
	?debugMsg(io_lib:format("~n Gradient:~p",[Grad])).	