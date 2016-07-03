function [y1,y2] = thresh_linear_sigmoid(y0, w1, b1, w2, b2, ...
                                         dropout_rate)
  if(~iscell(w1))
    w1 = {w1};
    b1 = {b1};
  end
  nl = length(w1);
  y1 = cell(1, nl);

  y1{1} = thresh_linear(bsxfun(@plus, w1{1}'*y0, b1{1}));
  if(dropout_rate(1) > 0)
    y1{1} = y1{1} .* (rand(size(y1{1}))>dropout_rate(1));
  end
  dropout_correction = 1;
  if(dropout_rate(1) < 0)
    dropout_correction = 1+dropout_rate(1);
  end

  for i=2:nl
    y1{i} = thresh_linear(bsxfun(...
      @plus, w1{i}'*y1{i-1}*dropout_correction, b1{i}));
    if(dropout_rate(i) > 0)
      y1{i} = y1{i} .* (rand(size(y1{i}))>dropout_rate(i));
    end
    dropout_correction = 1;
    if(dropout_rate(i) < 0)
      dropout_correction = 1+dropout_rate(i);
    end
  end

  y2 = sigmoid(bsxfun(...
    @plus, w2'*y1{end}*dropout_correction, b2));
end
