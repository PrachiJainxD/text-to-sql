{
    logdir: "/app/logdir/glove_run",
    model_config: "/app/configs/wikisql/nl2code-wikisql_inference.jsonnet",
    model_config_args: {
        att: 0,
        data_path: '/app/data/wikisql/',
    },

    eval_name: "wikisql_glove_run_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "/app/logdir/glove_run/ie_dirs", #eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: false, #true
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000], #eval_steps: [1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100, 9500],
    eval_section: "test", #"val"
}
