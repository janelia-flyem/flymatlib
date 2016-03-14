function tbar_json_write(fn, locs, multi_flag)
% TBAR_JSON_WRITE(fn, locs, multi_flag) write tbars to json file
%   using Raveler synapse format
%
%   fn           json output filename
%   locs         4xN matrix, locs(1:3,:) coords, locs(4,:) conf
%   multi_flag   optional 1xN vector

  jfid = fopen(fn, 'wt');
  fprintf(jfid,'{\n  "data": [\n');
  tbarcheck = 0;
  
  if(~exist('multi_flag','var'))
    multi_flag = zeros(1,size(locs,2));
  end
  
  for jj=1:size(locs,2)
    % no coordinate changing
    my=locs(1,jj);
    mx=locs(2,jj);
    mz=locs(3,jj);
    
    if tbarcheck
      fprintf(jfid,',\n    { \n');
    else
      fprintf(jfid,'    { \n');
    end
    
    fprintf(jfid,'\t "T-bar": { \n');
    fprintf(jfid,'\t\t"status": "working", \n');
    if(multi_flag(jj))
      fprintf(jfid,'\t\t"multi": "multi", \n');
    end
    fprintf(jfid,'\t\t"confidence": %g, \n', locs(4,jj));
    fprintf(jfid,'\t\t"body ID": -1, \n');
    fprintf(jfid,'\t\t"location": [ \n');
    fprintf(jfid,'\t\t   %d, \n', my);
    fprintf(jfid,'\t\t   %d, \n', mx);
    fprintf(jfid,'\t\t   %d  \n', mz);
    fprintf(jfid,'\t\t ] \n');
    fprintf(jfid,'\t }, \n');
    fprintf(jfid,'\t"partners": [  ]\n');
    fprintf(jfid,'    } ');
    
    tbarcheck=1;
  end
  fprintf(jfid,'\n  ],\n');
  fprintf(jfid,'  "metadata": {\n');
  fprintf(jfid,'\t  "username": "dummy",\n');
  fprintf(jfid,'\t  "software version": "dummy",\n');
  fprintf(jfid,'\t  "description": "synapse annotations",\n');
  fprintf(jfid,'\t  "file version": 1, \n');
  fprintf(jfid,'\t  "software version": "dummy",\n');
  fprintf(jfid,'\t  "computer": "dummy",\n');
  fprintf(jfid,'\t  "date": "dummy",\n');
  fprintf(jfid,'\t  "session path": "dummy",\n');
  fprintf(jfid,'\t  "software": "dummy"\n');
  fprintf(jfid,'  }\n}');
  
  fclose(jfid);

end
