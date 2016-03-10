function f_out = fml_qsub(f, pe_batch_size, use_gpu, ...
                          use_db_limit, use_short, ...
                          dfeval_dir, run_script, ...
                          varargin)
% FML_QSUB run compiled code distributed across SGE cluster
% f_out = FML_QSUB(f, pe_batch_size, use_gpu, ...
%                  use_db_limit, use_short, ...
%                  dfeval_dir, run_script, varargin)
%
%   f               function to run
%   pe_batch_size   number of slots per worker
%   use_gpu         set flag for gpu queue
%   use_db_limit    set flag for limit on concurrent workers
%   use_short       set flag to use short queue (<1 hour jobs)
%   dfeval_dir      directory for temporary files
%   run_script      script to run executable, see fml_get_exe.m
%   varargin        input arguments to f above


  qsub_loc = '/sge/8.2.1/bin/lx-amd64/qsub';

  if(~exist('pe_batch_size','var') || isempty(pe_batch_size))
    pe_batch_size = 1;
  end
  if(~exist('use_gpu','var') || isempty(use_gpu))
    use_gpu = 0;
  end
  if(~exist('use_db_limit','var') || isempty(use_db_limit))
    use_db_limit = 0;
  end
  if(~exist('use_short','var') || isempty(use_short))
    use_short = false;
  end
  assert(exist('dfeval_dir','var')>0 && ~isempty(dfeval_dir), ...
         'FML:AssertionFailed', ...
         ['directory to store temporary files for distributed ' ...
          'computing was not set']);

  use_gpu_string = ''; %#ok<NASGU>
  if(use_gpu > 0)
    use_gpu_string = '-l gpu_k20=true ';
    pe_batch_size  = 16; % must use all slots on machine
  else
    use_gpu_string = '';
  end
  use_db_limit_string = '';
  if(use_db_limit)
    use_db_limit_string = '-l limit_500=1 ';
  end
  use_short_string = '';
  if(use_short)
    use_short_string = '-l short=true ';
  end
  
  assert(exist(run_script,'file')>0, ...
         'FML:AssertionFailed', ...
         sprintf(...
           'script to run compiled executable %s does not exist', ...
           run_script));
  
  max_tasks_at_once = 3000;
  
  temp_dir = dfeval_dir;
  assert(exist(temp_dir,'dir')~=0, ...
         'FML:AssertionFailed', ...
         sprintf('directory %s does not exist', temp_dir));

  while(true)
    job_dir  = sprintf('%s/job_%s_%s', temp_dir, datestr(now,30), ...
                       get_random_id([],1));
    mkdir_st = system(sprintf('mkdir %s', job_dir));
    if(mkdir_st == 0)
      break
    end
  end
  fprintf('[job_dir=%s]\n', job_dir);
  [~,job_dir_end] = fileparts(job_dir);
  
  num_tasks = length(varargin{1});
  num_args  = length(varargin);
  for i=2:num_args
    arg_len = length(varargin{i});
    if(num_tasks == 1)
      num_tasks = arg_len;
    else
      assert(arg_len == num_tasks || arg_len == 1, ...
             'FML:AssertionFailed', ...
             'arguments not of same length');
    end
  end

  fn_in  = cell(1,num_tasks);
  fn_out = cell(1,num_tasks);
  
  for t=1:num_tasks
    bundle = struct('f',f);
    bundle.args = cell(1,num_args);
    for i=1:num_args
      if(length(varargin{i}) == 1)
        bundle.args{i} = varargin{i}{1};
      else
        bundle.args{i} = varargin{i}{t};
      end
    end
    
    fn_in{t}  = {sprintf('%s/task_%d.mat', job_dir, t)};
    fn_out{t} = {sprintf('%s/task_%d_return.mat', job_dir, t)};
    n_whos=whos('bundle');
    if(n_whos.bytes>(2e9))
      save(fn_in{t}{:}, 'bundle', '-v7.3');
    else
      save(fn_in{t}{:}, 'bundle');
    end
  end
  
  log_dir = sprintf('%s/log', job_dir);
  if(~exist(log_dir, 'dir'))
    system(sprintf('mkdir %s', log_dir));
  end

  % for t=1:num_tasks %#ok<UNRCH>
  %   cmd = sprintf(['%s %s %s/task_%d.mat ' ...
  %                  '>& %s/task_log_%d '], ...
  %                 run_script, get_random_id([],1), job_dir, t, ...
  %                 log_dir, t);
  %   fprintf('%s\n', cmd);
  %   system(cmd);
  %   % fn_out{t} = {dfeval_gbh_worker(fn_in{t}{:})};
  % end

  pe_batch_string = '';
  if(pe_batch_size > 1)
    pe_batch_string = sprintf('-pe batch %d ', pe_batch_size);
  end
  for task_batch = 1:max_tasks_at_once:num_tasks
    tb_upper = min(task_batch+max_tasks_at_once-1, num_tasks);

    log_file = sprintf('%s/array_log_%d', log_dir, ...
                       task_batch);

    single_task_to_run = 0;
    if(num_tasks > 1)
      cmd = sprintf(['%s ' ...
                     '-t %d-%d -N %s %s%s%s%s' ...
                     '-j y -o /dev/null ' ...
                     '-b y -cwd -sync y -S /bin/bash ' ...
                     '''%s ${SGE_TASK_ID}_%s ' ...
                     '%s/task_${SGE_TASK_ID}.mat ' ...
                     '>& %s/task_log_${SGE_TASK_ID}'' ' ...
                     '> %s'], ...
                    qsub_loc, ...
                    task_batch, tb_upper, ...
                    char(f), pe_batch_string, use_gpu_string, ...
                    use_db_limit_string, use_short_string, ...
                    run_script, get_random_id([],1), job_dir, ...
                    log_dir, log_file);
    else
      single_task_to_run = 1;
      cmd = sprintf(['%s ' ...
                     '-N %s %s%s%s%s' ...
                     '-j y -o /dev/null ' ...
                     '-b y -cwd -sync y -S /bin/bash ' ...
                     '''%s 1_%s ' ...
                     '%s/task_1.mat ' ...
                     '>& %s/task_log_1'' ' ...
                     '> %s'], ...
                    qsub_loc, ...
                    char(f), pe_batch_string, use_gpu_string, ...
                    use_db_limit_string, use_short_string, ...
                    run_script, get_random_id([],1), job_dir, ...
                    log_dir, log_file);
      
    end
    % add back:
    %   -V ?
    
    fprintf('%s\n', cmd);
    cmd_status = system(cmd); %#ok<NASGU>

    %% try some error handling instead of quitting
    num_repeat = 5;
    bad_idx_old = task_batch:tb_upper;
    
    for nr = 1:num_repeat+1
      java.lang.Thread.sleep(5*1e3);
      
      bad_idx_new = [];
      
      for tc = bad_idx_old
        if(~exist(fn_out{tc}{:}, 'file'))
          % keyboard
          fprintf('%d task failed, retrying...\n', tc);
          
          % save error output
          failed_log_file = ...
              sprintf('%s/task_log_%d', log_dir, tc);
          if(exist(failed_log_file, 'file'))
            system(sprintf( ...
              'cp %s %s/%s_task_log_%d_fail_%d', ...
              failed_log_file, dfeval_dir, job_dir_end, tc, ...
              nr-1));
          end
          
          bad_idx_new(end+1) = tc; %#ok<AGROW>
        end
      end
      
      if(~isempty(bad_idx_new) && nr <= num_repeat)
        tc_min = min(bad_idx_new);
        tc_max = max(bad_idx_new);

        if(~single_task_to_run)
          if(tc_min == tc_max)
            % avoid potential bug with SGE
            if(tc_min == 1)
              tc_max = 2;
            else
              tc_min = tc_max - 1;
            end
          end
          
          cmd = sprintf(['%s ' ...
                         '-t %d-%d -N %s %s%s%s%s' ...
                         '-j y -o /dev/null ' ...
                         '-b y -cwd -sync y -S /bin/bash ' ...
                         '''%s ${SGE_TASK_ID}_%s ' ...
                         '%s/task_${SGE_TASK_ID}.mat ' ...
                         '>& %s/task_log_${SGE_TASK_ID}'' ' ...
                         '> %s'], ...
                        qsub_loc, ...
                        tc_min, tc_max, ...
                        char(f), pe_batch_string, ...
                        use_gpu_string, use_short_string, ...
                        use_db_limit_string, ...
                        run_script, get_random_id([],1), ...
                        job_dir, ...
                        log_dir, log_file);
        else 
          cmd = sprintf(['%s ' ...
                         '-N %s %s%s%s%s' ...
                         '-j y -o /dev/null ' ...
                         '-b y -cwd -sync y -S /bin/bash ' ...
                         '''%s 1_%s ' ...
                         '%s/task_1.mat ' ...
                         '>& %s/task_log_1'' ' ...
                         '> %s'], ...
                        qsub_loc, ...
                        char(f), pe_batch_string, ...
                        use_gpu_string, use_short_string, ...
                        use_db_limit_string, ...
                        run_script, get_random_id([],1), ...
                        job_dir, ...
                        log_dir, log_file);
        end
        fprintf('%s\n', cmd);
        cmd_status = system(cmd); %#ok<NASGU>
      end
      
      bad_idx_old = bad_idx_new;
    end
    
    assert(isempty(bad_idx_old), 'JANCOM:AssertionFailed', ...
           'qsub failed');
    
  end

  try
    if(nargout(f) > 0)
      f_out = cell(num_tasks,1);
      for t=1:num_tasks
        if(~isempty(fn_out{t}) && ~isempty(fn_out{t}{:}))
          tmp_out  = load(fn_out{t}{:});
          f_out{t} = tmp_out.f_out(:)';
        end
      end
    end    
  catch exception
    exception %#ok<NOPRT>
    error('qsub return val error');
  end

  system(sprintf('rm -r %s', job_dir));

end
