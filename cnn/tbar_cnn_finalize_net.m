function net = tbar_cnn_finalize_net(net)
% net = TBAR_CNN_FINALIZE_NET(net)
% remove/replace loss layer as appropriate

% fix last layer
  if(strcmp(net.layers{end}.type, 'sigentloss'))
    % replace loss layer with sigmoid
    net.layers(end) = [];
    net.layers{end+1} = struct('type','sigmoid');
  end
  if(strcmp(net.layers{end}.type, 'squaredloss'))
    net.layers(end) = [];
  end

end
