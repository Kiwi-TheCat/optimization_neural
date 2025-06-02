function [particles, velocities] = update_particles(particles, velocities, p_best, g_best, w, c1, c2)
    % Particle Swarm Update Step
    % Inputs:
    % - particles: [num_particles × dim] current positions
    % - velocities: [num_particles × dim] current velocities
    % - p_best: best-known position of each particle
    % - g_best: best-known global position
    % - w: inertia weight
    % - c1: cognitive coefficient
    % - c2: social coefficient

    [num_particles, dim] = size(particles);

    r1 = rand(num_particles, dim);
    r2 = rand(num_particles, dim);

    velocities = w * velocities + ...
                 c1 * r1 .* (p_best - particles) + ...
                 c2 * r2 .* (g_best - particles);

    particles = particles + velocities;
end
