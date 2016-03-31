function x = fml_nnobjout(x)
% FML_NNOBJOUT replacement for fml_nncombobjloss during inference
% calls vl_nnsigmoid on x(:,:,:,1,:), all other channels remain
% at original values
  
  x(:,:,:,1,:) = vl_nnsigmoid(x(:,:,:,1,:));
end
