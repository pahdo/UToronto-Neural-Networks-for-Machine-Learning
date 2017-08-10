function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

    % CD1 Version 3 - PA4 Q9
    visible_data = sample_bernoulli(visible_data)

    h = visible_state_to_hidden_probabilities(rbm_w, visible_data);

    hidden_sample = sample_bernoulli(h);

    data_goodness = configuration_goodness_gradient(visible_data, hidden_sample);

    vis = hidden_state_to_visible_probabilities(rbm_w, hidden_sample);

    visible_sample = sample_bernoulli(vis);

    h_again = visible_state_to_hidden_probabilities(rbm_w, visible_sample);

    data_goodness_again = configuration_goodness_gradient(visible_sample, h_again);
    ret = data_goodness .- data_goodness_again;   

    % CD1 Version 2 - PA4 Q8
    % h = visible_state_to_hidden_probabilities(rbm_w, visible_data);

    % hidden_sample = sample_bernoulli(h);

    % data_goodness = configuration_goodness_gradient(visible_data, hidden_sample);

    % vis = hidden_state_to_visible_probabilities(rbm_w, hidden_sample);

    % visible_sample = sample_bernoulli(vis);

    % h_again = visible_state_to_hidden_probabilities(rbm_w, visible_sample);

    % data_goodness_again = configuration_goodness_gradient(visible_sample, h_again);
    % ret = data_goodness .- data_goodness_again;   

    % CD1 Version 1 - PA4 Q7
    % h = visible_state_to_hidden_probabilities(rbm_w, visible_data);

    % hidden_sample = sample_bernoulli(h);

    % data_goodness = configuration_goodness_gradient(visible_data, hidden_sample);

    % vis = hidden_state_to_visible_probabilities(rbm_w, hidden_sample);

    % visible_sample = sample_bernoulli(vis);

    % h_again = visible_state_to_hidden_probabilities(rbm_w, visible_sample);

    % h_again_sample = sample_bernoulli(h_again);

    % data_goodness_again = configuration_goodness_gradient(visible_sample, h_again_sample);
    % ret = data_goodness .- data_goodness_again;
end
