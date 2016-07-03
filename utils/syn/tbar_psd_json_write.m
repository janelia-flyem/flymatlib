function tbar_psd_json_write(fn, locs, plocs)

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
    if(size(locs,1)>4)
      fprintf(jfid,'\t\t"body ID": %d, \n', locs(5,jj));
    else
      fprintf(jfid,'\t\t"body ID": -1, \n');
    end
    fprintf(jfid,'\t\t"location": [ \n');
    fprintf(jfid,'\t\t   %d, \n', my);
    fprintf(jfid,'\t\t   %d, \n', mx);
    fprintf(jfid,'\t\t   %d  \n', mz);
    fprintf(jfid,'\t\t ] \n');
    fprintf(jfid,'\t }, \n');
    
    n_psd = size(plocs{jj},2);
    if(n_psd==0)
      fprintf(jfid,'\t"partners": [  ]\n');
    else
      fprintf(jfid,'\t"partners": [\n');
      for kk=1:n_psd
        fprintf(jfid,'\t\t{\n');

        fprintf(jfid,'\t\t\t"confidence": %g, \n', ...
                plocs{jj}(4,kk));
        if(size(plocs{jj},1)>4)
          fprintf(jfid,'\t\t\t"body ID": %d, \n', plocs{jj}(5,kk));
        else
          fprintf(jfid,'\t\t\t"body ID": -1, \n');
        end
        fprintf(jfid,'\t\t\t"location": [ \n');
        fprintf(jfid,'\t\t\t   %d, \n', plocs{jj}(1,kk));
        fprintf(jfid,'\t\t\t   %d, \n', plocs{jj}(2,kk));
        fprintf(jfid,'\t\t\t   %d  \n', plocs{jj}(3,kk));
        fprintf(jfid,'\t\t\t ] \n');        
        
        if(kk==n_psd)
          fprintf(jfid,'\t\t}\n');
        else
          fprintf(jfid,'\t\t},\n');
        end
      end
      fprintf(jfid,'\t]\n');
    end
    
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
