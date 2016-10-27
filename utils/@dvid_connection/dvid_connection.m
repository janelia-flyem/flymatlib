classdef dvid_connection

  properties
    machine_name
    repo_name
    user_name
    app_name

    use_httpie = 1
  end

  methods
    function obj = dvid_connection(machine_name, repo_name, ...
                                   user_name, app_name)
      obj.machine_name = machine_name;
      obj.repo_name    = repo_name;
      obj.user_name    = user_name;
      obj.app_name     = app_name;
    end

    function s = user_string(this)
      s = sprintf('u=%s&app=%s', this.user_name, this.app_name);
    end

    function s = http_cmd(this)
      if(this.use_httpie)
        s = 'http --check-status';
      else
        s = 'curl -s -f -X';
      end
    end

    function run_dvid_cmd(this, dvid_cmd)
      max_tries = 5;
      st = system(dvid_cmd);
      num_tries = 1;
      while(st ~= 0 && num_tries < max_tries)
        fprintf('status code = %d ,', st);
        if(st ~= 5) % not 503 unavailable from throttling
          num_tries = num_tries+1;
        end
        pause(30 + 30*rand());

        fprintf('retrying...\n');
        st = system(dvid_cmd);
      end
      if(st ~= 0)
        error('error connecting to dvid: %s/%s/', ...
              this.machine_name, this.repo_name);
      end
    end

    function branch(this, note)
      dvid_cmd = sprintf(...
          ['curl -s -f -X POST %s/api/node/%s/branch ' ...
           '-d ''{"note": "%s"}'''], ...
          this.machine_name, this.repo_name, note);
      st = system(dvid_cmd);

      if(st ~= 0)
        error('error connecting to dvid: %s', dvid_cmd);
      end
    end

    function commit(this, note)
      dvid_cmd = sprintf(...
          ['curl -s -f -X POST %s/api/node/%s/commit ' ...
           '-d ''{"note": "%s"}'''], ...
          this.machine_name, this.repo_name, note);
      st = system(dvid_cmd);

      if(st ~= 0)
        error('error connecting to dvid: %s', dvid_cmd);
      end
    end

    function create_instance(this, typename, dataname)
      assert(ismember(typename, ...
                      {'annotation', 'labelblk', 'labelvol'}), ...
             'FML:AssertionFailed', ...
             sprintf('unknown typename: %s', typename));

      dvid_cmd = sprintf(...
          ['curl -s -f -X POST %s/api/repo/%s/instance ' ...
           '-d ''{"typename": "%s", "dataname": "%s"}'''], ...
          this.machine_name, this.repo_name, ...
          typename, dataname);
      st = system(dvid_cmd);

      if(st ~= 0)
        error('error connecting to dvid: %s', dvid_cmd);
      end
    end

    function sync(this, dataname, syncname, do_reverse)
      dvid_cmd = sprintf(...
          ['curl -s -f -X POST %s/api/node/%s/%s/sync ' ...
           '-d ''{"sync": "%s"}'''], ...
          this.machine_name, this.repo_name, ...
          dataname, syncname);
      st = system(dvid_cmd);

      if(st ~= 0)
        error('error connecting to dvid: %s', dvid_cmd);
      end

      if(exist('do_reverse','var') && do_reverse)
        this.sync(syncname, dataname);
      end
    end


    % defined externally
    [im_mean, im_std, empty_vol] = get_image(...
        this, vol_start, vol_sz, image_fn, ...
        do_normalize, bg_vals_to_nan)
    seg = get_segmentation(...
        this, vol_start, vol_sz, seg_fn, seg_name);
    set_segmentation(this, vol_offset, ...
                           seg_fn, seg_name, ...
                           chunk_sz, do_permute)
    [tt,pp] = get_annotations(this, vol_start, vol_sz, ...
                                    annotations_name, ...
                                    is_groundtruth, bf, ...
                                    exclude_multi)
    [ll_t, ll_p] = get_labels(this, seg_name, tt_pts, pp_pts)
  end

end
