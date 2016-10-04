function net = tbar_cnn_finalize_net(net, remove_center_out)
% net = TBAR_CNN_FINALIZE_NET(net, remove_center_out)
% remove/replace loss layer as appropriate
%
% remove_center_out
%   remove last 3 channels of layer before objloss (default false)

  if(~exist('remove_center_out','var') || ...
     isempty(remove_center_out))
    remove_center_out = false;
  end

% fix last layer
  if(strcmp(net.layers{end}.type, 'sigentloss'))
    % replace loss layer with sigmoid
    net.layers(end) = [];
    net.layers{end+1} = struct('type','sigmoid');
  end
  if(strcmp(net.layers{end}.type, 'squaredloss') || ...
     strcmp(net.layers{end}.type, 'l1loss'))
    net.layers(end) = [];
  end
  if(strcmp(net.layers{end}.type, 'objloss'))
    if(remove_center_out)
      % remove extra output layers
      net.layers{end-1}.weights{1} = ...
          net.layers{end-1}.weights{1}(:,:,:,:,1);
      net.layers{end-1}.weights{2} = ...
          net.layers{end-1}.weights{2}(:,1);
      % replace loss layer with sigmoid
      net.layers(end) = [];
      net.layers{end+1} = struct('type','sigmoid');
    else
      % replace loss layer with objout
      net.layers(end) = [];
      net.layers{end+1} = struct('type','objout');
    end
  end
end
