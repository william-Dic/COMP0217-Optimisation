classdef EKF_Localization < handle % Use handle class for mutable object state
    properties
        x % State vector [x; y; yaw] (3x1 column vector)
        P % Covariance matrix (3x3)
        % Q % Process noise covariance - NOW CALCULATED IN PREDICT
        R % Measurement noise covariance (3x3)

        % --- Parameters for myEKF-style Prediction ---
        current_v_est       % Current estimated forward velocity
        gyro_bias_z         % Bias for Z-axis gyroscope
        accel_bias_fwd      % Bias for forward accelerometer
        process_noise_pos   % Std dev for position process noise component
        process_noise_yaw_rate % Std dev for yaw rate process noise component (rad/s)
        process_noise_accel % Std dev for acceleration process noise component

        % --- Arena and Sensor Geometry ---
        arena_limits
        tof_rel_pos   % 2x3 matrix [[x1;y1], [x2;y2], [x3;y3]] relative offsets
        tof_rel_angle % 1x3 vector [angle1, angle2, angle3] relative angles
        epsilon
    end

    methods
        % Modified Constructor to accept parameters for the new prediction model
        function obj = EKF_Localization(initial_state, initial_covariance, ...
                                        ekf_params, ... % Struct containing biases and process noises
                                        measurement_noise_r_std)
            % Constructor
            obj.x = double(initial_state(:)); % Ensure column vector and double precision
            obj.P = double(initial_covariance);

            % --- Initialize Parameters for Prediction ---
            % Use defaults similar to myEKF if not provided
            if isfield(ekf_params, 'initial_velocity')
                obj.current_v_est = ekf_params.initial_velocity;
            else
                obj.current_v_est = 0.1; % Default initial velocity guess
                warning('Initial velocity not provided, defaulting to %.2f m/s', obj.current_v_est);
            end
             if isfield(ekf_params, 'gyro_bias_z')
                % obj.gyro_bias_z = ekf_params.gyro_bias_z;
                obj.gyro_bias_z = 0.0015;
            else
                obj.gyro_bias_z = 0.0015; % Default from myEKF
                warning('Gyro bias Z not provided, defaulting to %.4f rad/s', obj.gyro_bias_z);
            end
             if isfield(ekf_params, 'accel_bias_fwd')
                obj.accel_bias_fwd = ekf_params.accel_bias_fwd;
            else
                obj.accel_bias_fwd = 0.006; % Default from myEKF
                warning('Accel bias Fwd not provided, defaulting to %.4f m/s^2', obj.accel_bias_fwd);
            end
            if isfield(ekf_params, 'process_noise_pos')
                obj.process_noise_pos = ekf_params.process_noise_pos;
            else
                obj.process_noise_pos = 0.5; % Default from myEKF
                warning('Process noise Pos not provided, defaulting to %.2f m/s', obj.process_noise_pos);
            end
            if isfield(ekf_params, 'process_noise_yaw_rate')
                obj.process_noise_yaw_rate = ekf_params.process_noise_yaw_rate;
            else
                obj.process_noise_yaw_rate = (1 * pi/180); % Default from myEKF (1 deg/s)
                warning('Process noise Yaw Rate not provided, defaulting to %.4f rad/s', obj.process_noise_yaw_rate);
            end
            if isfield(ekf_params, 'process_noise_accel')
                obj.process_noise_accel = ekf_params.process_noise_accel;
            else
                obj.process_noise_accel = 0.5; % Default from myEKF
                warning('Process noise Accel not provided, defaulting to %.2f m/s^2', obj.process_noise_accel);
            end

            % --- Initialize Measurement Noise Covariance R ---
             if length(measurement_noise_r_std) == 3
                obj.R = diag(double(measurement_noise_r_std(:)).^2);
             else
                % If only one std dev provided, assume all ToF are the same
                if isscalar(measurement_noise_r_std)
                    obj.R = diag(repmat(double(measurement_noise_r_std)^2, 3, 1));
                    warning('Single measurement_noise_r_std provided; assuming identical noise for all 3 ToF sensors.');
                else
                     error('measurement_noise_r_std must have 1 or 3 elements');
                end
             end

            % --- Arena and Sensor Geometry ---
            arena_width = 2.4;
            arena_height = 2.4;
            obj.arena_limits = struct(...
                'x_min', -arena_width / 2.0, ...
                'x_max',  arena_width / 2.0, ...
                'y_min', -arena_height / 2.0, ...
                'y_max',  arena_height / 2.0 ...
            );
            obj.tof_rel_pos = double([
                 0.01,  -0.27, -0.01; % x-offsets relative to robot center
                 0.0,  0.01,  0.01  % y-offsets relative to robot center
            ]);
            obj.tof_rel_angle = double([
                pi/2, pi, -pi/2 % Angle offsets relative to robot yaw
            ]);
            obj.epsilon = 1e-6; % Small number

            disp('EKF Initialized (with myEKF Prediction Logic):');
            fprintf('  Initial State (x): [%.3f, %.3f, %.3f rad]\n', obj.x(1), obj.x(2), obj.x(3));
            disp('  Initial Covariance (P):'); disp(obj.P);
            fprintf('  Gyro Bias Z: %.4f, Accel Bias Fwd: %.4f\n', obj.gyro_bias_z, obj.accel_bias_fwd);
            fprintf('  Process Noise Std Devs (Pos:%.2f, YawRate:%.4f, Accel:%.2f)\n', obj.process_noise_pos, obj.process_noise_yaw_rate, obj.process_noise_accel);
            disp('  Measurement Noise (R):'); disp(obj.R);
        end

        % ================================================================
        % === PREDICT METHOD: REPLACED WITH myEKF LOGIC ===
        % ================================================================
        % Now requires raw gyro and accelerometer readings as input
        function predict(obj, gyro_z_raw, accel_fwd_raw, dt)
            % Predicts state using gyro for yaw rate and accelerometer for velocity estimation.
            % Adapted from the logic in myEKF.

            if dt <= 0
                warning('dt is non-positive (%.4f) in predict step. Skipping prediction.', dt);
                return;
            end

            % --- Previous State ---
            x_prev = obj.x;
            P_prev = obj.P;
            yaw_prev = x_prev(3); % Already normalized from previous step or init

            % --- Process Inputs ---
            % Correct Gyro Reading
            omega_z_corrected = gyro_z_raw - obj.gyro_bias_z;

            % Correct Accelerometer Reading and Update Velocity Estimate
            accel_fwd_corrected = accel_fwd_raw - obj.accel_bias_fwd;
            obj.current_v_est = obj.current_v_est + accel_fwd_corrected * dt; % Integrate acceleration
            % Optional: Add damping or limits to velocity estimate if needed
            % e.g., obj.current_v_est = obj.current_v_est * 0.99;

            % --- State Transition Model f(x, u) ---
            % Predict next state based on previous state, corrected inputs, and estimated velocity
            x_pred = zeros(3,1);
            x_pred(1) = x_prev(1) + obj.current_v_est * cos(yaw_prev) * dt;
            x_pred(2) = x_prev(2) + obj.current_v_est * sin(yaw_prev) * dt;
            x_pred(3) = yaw_prev + omega_z_corrected * dt;
            x_pred(3) = EKF_Localization.normalizeAngle(x_pred(3)); % Normalize predicted angle

            % --- State Transition Jacobian F ---
            % Jacobian of f w.r.t. x = [x, y, yaw]
            F = [ 1  0  -obj.current_v_est * sin(yaw_prev) * dt;
                  0  1   obj.current_v_est * cos(yaw_prev) * dt;
                  0  0   1 ];

            % --- Process Noise Covariance Q_k ---
            % Calculate the process noise covariance for this time step dt
            % This matches the structure in myEKF where Q depends on dt and noise parameters
            % Term related to uncertainty propagation due to acceleration noise
            Q_pos_from_accel = (0.5 * obj.process_noise_accel * dt^2)^2;

            % Combine position noise (from base process noise + accel uncertainty)
            % and yaw rate noise
            Q_k = diag([ (obj.process_noise_pos * dt)^2 + Q_pos_from_accel, ...
                         (obj.process_noise_pos * dt)^2 + Q_pos_from_accel, ...
                         (obj.process_noise_yaw_rate * dt)^2 ]);

            % --- Predict Covariance ---
            obj.P = F * P_prev * F' + Q_k;

            % Ensure P remains symmetric (optional but good practice)
             obj.P = 0.5 * (obj.P + obj.P');

            % --- Update State ---
            obj.x = x_pred;

        end % End predict method


        % ================================================================
        % === UPDATE METHOD: UNCHANGED ===
        % ================================================================
        % ================================================================
        % === UPDATE METHOD: MODIFIED FOR DEBUGGING YAW ===
        % ================================================================
        function update(obj, z)
            z = double(z(:));

            h_x = obj.compute_measurement_model_h(obj.x);
            H = obj.compute_numerical_jacobian_H(obj.x); % Calculate full Jacobian

            % --- <<< ADD THIS FOR DEBUGGING >>> ---
            % Temporarily disable ToF updates for the Yaw state (state index 3)
            % Set the 3rd column of H (dh/dyaw) to zero before calculating gain
            disable_tof_yaw_update = true; % Set to false to re-enable
            if disable_tof_yaw_update
                H_modified = H;
                H_modified(:, 3) = 0; % Zero out sensitivity to yaw
                fprintf('DEBUG: ToF Yaw update DISABLED.\n'); % Add message
            else
                H_modified = H; % Use original Jacobian
                 fprintf('DEBUG: ToF Yaw update ENABLED.\n'); % Add message
            end
            % --- <<< END OF DEBUGGING MODIFICATION >>> ---


            if all(H_modified == 0, 'all') % Use modified H here
                 warning('Measurement Jacobian H (potentially modified) is all zeros. Skipping update.');
                 return;
            end

            % Use H_modified for Kalman Gain calculation
            S = H_modified * obj.P * H_modified' + obj.R; % Use modified H

            if cond(S) > 1/eps
                warning('Innovation covariance matrix S is singular or ill-conditioned (cond=%.2e). Skipping update.', cond(S));
                return;
            end

            K = obj.P * H_modified' / S; % Use modified H

            y = z - h_x;
            obj.x = obj.x + K * y; % Update state (Yaw won't be affected if H_modified(:,3) was 0)
            obj.x(3) = EKF_Localization.normalizeAngle(obj.x(3));

            % Joseph form update (use original H here for theoretical correctness,
            % although using H_modified might be more consistent with the gain used)
            % Let's use H_modified for consistency with K:
            I = eye(size(obj.P,1));
            obj.P = (I - K * H_modified) * obj.P * (I - K * H_modified)' + K * obj.R * K';

            obj.P = 0.5 * (obj.P + obj.P');
            [V, D] = eig(obj.P);
            D(D < 0) = eps;
            obj.P = V * D * V';
        end

        % --- Helper Methods (UNCHANGED) ---
        function h = compute_measurement_model_h(obj, state)
            % Computes the expected measurement vector h(x) for all sensors.
            h = zeros(3, 1);
            for i = 1:3
                h(i) = obj.calculate_expected_tof_single(state, i);
            end
        end

        function dist = calculate_expected_tof_single(obj, state, sensor_index)
            % Calculates the expected distance for a single ToF sensor using ray casting.
             x_r = state(1);
            y_r = state(2);
            yaw_r = state(3); % Already normalized in predict/update

            rel_pos_sensor = obj.tof_rel_pos(:, sensor_index); % [x_rel; y_rel]
            rel_angle_sensor = obj.tof_rel_angle(sensor_index);

            % Calculate sensor's global position
            c_yaw = cos(yaw_r);
            s_yaw = sin(yaw_r);
            rot_matrix = [c_yaw, -s_yaw; s_yaw, c_yaw];
            sensor_offset_world = rot_matrix * rel_pos_sensor;
            sx_global = x_r + sensor_offset_world(1);
            sy_global = y_r + sensor_offset_world(2);

            % Calculate sensor's global orientation (angle)
            s_yaw_global = EKF_Localization.normalizeAngle(yaw_r + rel_angle_sensor);
            c_s_yaw = cos(s_yaw_global);
            s_s_yaw = sin(s_yaw_global);

            % Calculate distances to walls along the sensor's ray
            distances = [];
            limits = obj.arena_limits;

            % Right wall (x = x_max)
            if abs(c_s_yaw) > obj.epsilon % Check if ray is not parallel to wall
                d = (limits.x_max - sx_global) / c_s_yaw;
                py = sy_global + d * s_s_yaw; % Intersection y-coord
                if d > obj.epsilon && py >= limits.y_min && py <= limits.y_max
                    distances = [distances, d];
                end
            end
            % Left wall (x = x_min)
            if abs(c_s_yaw) > obj.epsilon
                d = (limits.x_min - sx_global) / c_s_yaw;
                py = sy_global + d * s_s_yaw;
                if d > obj.epsilon && py >= limits.y_min && py <= limits.y_max
                    distances = [distances, d];
                 end
            end
            % Top wall (y = y_max)
            if abs(s_s_yaw) > obj.epsilon % Check if ray is not parallel to wall
                d = (limits.y_max - sy_global) / s_s_yaw;
                px = sx_global + d * c_s_yaw; % Intersection x-coord
                if d > obj.epsilon && px >= limits.x_min && px <= limits.x_max
                    distances = [distances, d];
                 end
            end
            % Bottom wall (y = y_min)
            if abs(s_s_yaw) > obj.epsilon
                d = (limits.y_min - sy_global) / s_s_yaw;
                px = sx_global + d * c_s_yaw;
                if d > obj.epsilon && px >= limits.x_min && px <= limits.x_max
                     distances = [distances, d];
                 end
            end

            % Return the minimum valid distance
            if isempty(distances)
                warning('EKF:NoWallHit', 'Sensor %d ray did not hit any wall segments. State: [%.2f, %.2f, %.2f rad]', sensor_index, x_r, y_r, yaw_r);
                dist = 10.0; % Return a large distance
            else
                dist = min(distances);
            end
        end

        function H = compute_numerical_jacobian_H(obj, state, delta)
            % Computes the measurement Jacobian H numerically using central differences.
            if nargin < 3
                delta = 1e-7;
            end
            H = zeros(3, 3);
            for j = 1:3
                x_plus = state; x_minus = state;
                x_plus(j) = x_plus(j) + delta;
                x_minus(j) = x_minus(j) - delta;
                 if j == 3 % Normalize yaw if perturbed
                    x_plus(3) = EKF_Localization.normalizeAngle(x_plus(3));
                    x_minus(3) = EKF_Localization.normalizeAngle(x_minus(3));
                 end
                h_plus = obj.compute_measurement_model_h(x_plus);
                h_minus = obj.compute_measurement_model_h(x_minus);
                derivative = (h_plus - h_minus) / (2 * delta);
                H(:, j) = derivative;
            end
        end

    end % end methods

    methods (Static)
        function normalized = normalizeAngle(angle)
            % Normalize an angle to the range [-pi, pi].
            % Equivalent to wrapToPi
            normalized = mod(angle + pi, 2*pi) - pi;
            % Alternative: normalized = atan2(sin(angle), cos(angle));
        end
    end % end static methods

end % end classdef