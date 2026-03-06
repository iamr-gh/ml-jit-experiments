const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const exe = b.addExecutable(.{
        .name = "zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const nn_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/nn.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const tensor_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tensor.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_main_tests = b.addRunArtifact(main_tests);
    const run_nn_tests = b.addRunArtifact(nn_tests);
    const run_tensor_tests = b.addRunArtifact(tensor_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_main_tests.step);
    test_step.dependOn(&run_nn_tests.step);
    test_step.dependOn(&run_tensor_tests.step);
}
