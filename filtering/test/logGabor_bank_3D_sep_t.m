classdef logGabor_bank_3D_sep_t
    %LOGGABOR_BANK_3D_SEP_T 3D space-time separable log-Gabor bank
    
    properties
        input_size          = [256 512 50]
        num_scales          = 3
        orientations        = [0 60 120]
        num_orientations    = 3
        velocities          = [1 -1 1/3 -1/3 0]
        num_velocities      = 5
        min_wavelength      = 4
        mult                = 2.1
        sigma_on_f          = 0.55
        dtheta_max          = 60
        filters_spatial     = {}
        filters_temporal    = {}
        filter_energies     = []
        spacing             = 1
        spacing_spatial     = 1
        spacing_temporal    = 1
    end
    
    methods
        function obj = logGabor_bank_3D_sep_t()
        end
        
        function obj = set_up_filters(obj)
            %SET_UP_FILTERS
            obj.dtheta_max = min(obj.dtheta_max, obj.orientations(2) - obj.orientations(1));

            obj.num_orientations = numel(obj.orientations);
            obj.num_velocities = numel(obj.velocities);
            obj.filters_spatial = cell(obj.num_orientations, obj.num_scales);
            obj.filters_temporal = cell(obj.num_velocities, obj.num_scales);

            rows = obj.input_size(1);
            cols = obj.input_size(2);
            frames = obj.input_size(3);
            fx = single(reshape(ifftshift(fft_freqs(cols)), [1 cols 1]));
            fy = single(reshape(ifftshift(fft_freqs(rows)), [rows 1 1]));
            ft = single(reshape(ifftshift(fft_freqs(frames)), [1 1 frames]));
            f_mag_sp = hypot(fx, fy);
            f_theta = atan2d(fy, fx);
            sinftheta = sind(f_theta);
            cosftheta = cosd(f_theta);

            % low-pass Butterworth filter
            % these parameters won't need to be changed..
            lp_cutoff = 0.45;
            lp_n = 15;
            lp = 1.0 ./ (1.0 + (f_mag_sp ./ lp_cutoff).^(2*lp_n));

            for i1 = obj.num_scales:-1:1
                f0 = (1.0 / obj.min_wavelength) / obj.mult^(obj.num_scales-i1);

                % make spatial filters
                lg_r = exp((-(log(f_mag_sp/f0)).^2) / (2 * log(obj.sigma_on_f)^2));
                lg_r = lg_r.*lp;    % Apply low-pass filter
                lg_r(1,1) = 0;      % Set the value at the 0 frequency point of the filter
                                    % back to zero (undo the radius fudge).
                for i2 = 1:obj.num_orientations
                    angl = obj.orientations(i2);
                    ds_s = sinftheta * cosd(angl) - cosftheta * sind(angl);    % Difference in sine.
                    dc_s = cosftheta * cosd(angl) + sinftheta * sind(angl);    % Difference in cosine.
                    dftheta = abs(atan2d(ds_s,dc_s));                             % Absolute angular distance.
                    % Scale theta so that cosine spread function has the right wavelength and clamp to pi    
                    dftheta = 180*min(1,dftheta/obj.dtheta_max);
                    % The spread function is cosd(dtheta) between -180 and 180.  We add 1,
                    % and then divide by 2 so that the value ranges 0-1
                    spread = (cosd(dftheta)+1)/2;
                    obj.filters_spatial{i2,i1} = lg_r .* spread;
                end

                % temporal filters
                % -ve because that's the correct temporal frequency for
                % that speed (the filter construction below takes care to
                % implement this correctly)
                ft_c = -f0 * obj.velocities;
                for i2 = 1:obj.num_velocities
                    if ft_c(i2) == 0
                        % lowpass
                        ft_c_mag = abs(ft_c);
                        f1 = min(ft_c_mag(ft_c_mag > 0), [], 'all');
                        sigmaf = f1/2;
                        obj.filters_temporal{i2,i1} = exp(-0.5*(ft/sigmaf).^2);
                    elseif abs(ft_c(i2)) > (1/3)
                        fprintf(2, 'filter is vulnerable to aliasing --> skipping\n');
                        obj.filters_temporal{i2,i1} = [];
                    else
                        lg_ft = zeros(size(ft));
                        ft_scaled = ft / ft_c(i2);
                        lg_ft(ft_scaled > 0) = exp(-(log(ft_scaled(ft_scaled > 0)).^2) / (2*log(obj.sigma_on_f)^2));
                        obj.filters_temporal{i2,i1} = lg_ft;
                    end
                end
            end
            obj = obj.set_up_filter_energies();
        end

        function obj = set_up_filter_energies(obj)
            obj.num_orientations = numel(obj.orientations);
            obj.num_velocities = numel(obj.velocities);
            obj.filter_energies = zeros(obj.num_velocities*obj.num_orientations, obj.num_scales);
            for ind_s = 1:obj.num_scales
                for ind_filt = 1:obj.num_orientations*obj.num_velocities
                    [ind_v, ind_o] = obj.ind_filt_to_ind_tuning(ind_filt);
                    Gs = obj.filters_spatial{ind_o, ind_s};
                    Gt = obj.filters_temporal{ind_v, ind_s};
                    if ~isempty(Gt)
                        g = fftshift(ifftn(Gs .* Gt));
                        obj.filter_energies(ind_filt,ind_s) = sum(abs(g).^2, 'all');
                    end
                end
            end
        end

        function [ind_filt] = ind_tuning_to_ind_filt(obj, ind_v, ind_o)
            ind_filt = (ind_v - 1) * obj.num_orientations + ind_o;
        end

        function [ind_v, ind_o] = ind_filt_to_ind_tuning(obj, ind_filt)
            ind_v = 1 + floor((ind_filt-1)/obj.num_orientations);
            ind_o = 1 + mod((ind_filt-1), obj.num_orientations);
        end

        function [v, theta, f0] = tuning(obj, ind_filt, ind_scale)
            [ind_v, ind_o] = obj.ind_filt_to_ind_tuning(ind_filt);
            v = obj.velocities(ind_v);
            theta = obj.orientations(ind_o);
            if nargin < 3
                f0 = [];
            else
                f0 = obj.scaled_frequency(ind_scale);
            end
        end

        function [dirs] = tuning_directions(obj)
            % NOTE: these directions are in frequency-domain
            dirs = zeros(obj.num_orientations*obj.num_velocities, 3);
            for n = 1:obj.num_orientations*obj.num_velocities
                [v, thetad] = obj.tuning(n);
                phid = phid_vtuned(v);
                dirs(n,:) = dircosd_3D(phid, thetad);
            end
        end

        function [f0] = scaled_frequency(obj, ind_scale)
            f0 = 1.0 / (obj.min_wavelength * (obj.mult^(obj.num_scales - ind_scale)));
        end
        
        function [R, Rz] = response(obj, I, flux_noise_std)
            if nargin < 3
                flux_noise_std = [];
            end

            R = cell(obj.num_velocities*obj.num_orientations, obj.num_scales);
            Rz = cell(size(R));

            I_F_sp = fft2(I);
            for ind_s = 1:obj.num_scales
                R_sp = cell(obj.num_orientations, 1);
                for ind_o = 1:obj.num_orientations
                    G = obj.filters_spatial{ind_o,ind_s};
                    if isgpuarray(I_F_sp) && ~isgpuarray(G)
                        G = gpuArray(G);
                    end
                    R_sp{ind_o} = ifft2(I_F_sp .* G);
                    if isscalar(obj.spacing_spatial)
                        ss = obj.spacing_spatial;
                    else
                        ss = obj.spacing_spatial(ind_s);
                    end
                    if ss ~= 1
                        R_sp{ind_o} = R_sp{ind_o}(1+floor(ss/2):ss:end, 1+floor(ss/2):ss:end, :);
                    end
                    R_F_t = fft(R_sp{ind_o}, [], 3);
                    for ind_v = 1:obj.num_velocities
                        G = obj.filters_temporal{ind_v,ind_s};
                        if ~isempty(G) && ~((obj.velocities(ind_v) == 0) && obj.orientations(ind_o) >= 180)
                            if isgpuarray(R_F_t) && ~isgpuarray(G)
                                G = gpuArray(G);
                            end
                            ind_filt = obj.ind_tuning_to_ind_filt(ind_v, ind_o);
                            R{ind_filt,ind_s} = ifft(R_F_t .* G, [], 3);
                            if isscalar(obj.spacing_temporal)
                                st = obj.spacing_temporal;
                            else
                                st = obj.spacing_temporal(ind_v,ind_s);
                            end
                            if st ~= 1
                                R{ind_filt,ind_s} = R{ind_filt,ind_s}(:,:,1+floor(st/2):st:end);
                            end
                            if nargout >= 2
                                if isempty(flux_noise_std)
                                    Rz{ind_filt,ind_s} = inf;
                                else
                                    % calculate z-scores
                                    % for 1 component (real/imag) we want the
                                    % 0.5 factor
                                    noise_std_os = flux_noise_std;
                                    if any(size(noise_std_os) ~= size(R{ind_filt,ind_s}))
                                        noise_std_os = imresize3(noise_std_os, size(R{ind_filt,ind_s}), 'nearest');
                                    end
                                    std_R = stdev_response(noise_std_os, obj.filter_energies(ind_filt, ind_s));
                                    Rz{ind_filt,ind_s} = abs(R{ind_filt,ind_s}) ./ std_R;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
