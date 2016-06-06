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

    % defined externally
    [im_mean, im_std, empty_vol] = get_image(...
        this, vol_start, vol_sz, image_fn, ...
        do_normalize, bg_vals_to_nan)

  end

end
