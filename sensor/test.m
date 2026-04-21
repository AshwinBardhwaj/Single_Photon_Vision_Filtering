fprintf('=== ppd ===\n');
fprintf('ppd(0)    = %.6f\n', ppd(0));
fprintf('ppd(1)    = %.6f\n', ppd(1));
fprintf('ppd(5)    = %.6f\n', ppd(5));
fprintf('ppd(inf)  = %.6f\n', ppd(inf));
arr = ppd([0, 0.5, 1.0, 2.0, 5.0]);
fprintf('ppd array = [%s]\n', strjoin(compose('%g', arr), ' '));

fprintf('\n=== ippd ===\n');
fprintf('ippd(0)    = %.6f\n', ippd(0));
fprintf('ippd(0.5)  = %.6f\n', ippd(0.5));
fprintf('ippd(0.99) = %.6f\n', ippd(0.99));
arr = ippd([0, 0.25, 0.5, 0.75]);
fprintf('ippd array = [%s]\n', strjoin(compose('%g', arr), ' '));

fprintf('\n=== roundtrip ippd(ppd(x)) ===\n');
xs = [0.1, 1.0, 2.5, 5.0, 10.0];
fprintf('x            = [%s]\n', strjoin(compose('%g', xs), ' '));
fprintf('ippd(ppd(x)) = [%s]\n', strjoin(compose('%g', ippd(ppd(xs))), ' '));

fprintf('\n=== calc_bitdepth ===\n');
Ms_bd = [1, 2, 3, 4, 15, 16, 17, 255, 256, 1024];
for i = 1:length(Ms_bd)
    fprintf('calc_bitdepth(%d) = %d\n', Ms_bd(i), calc_bitdepth(Ms_bd(i)));
end

fprintf('\n=== binom_noise_std ===\n');
cs = [0.1, 0.5, 1.0, 2.0];
fprintf('c = [%s]\n', strjoin(compose('%g', cs), ' '));
for M = [1, 4, 16]
    std_vals = binom_noise_std(cs, M);
    fprintf('M=%2d, std = [%s]\n', M, strjoin(compose('%g', std_vals), ' '));
end

rng(42);

fprintf('\n=== quanta_sample_direct (statistical) ===\n');
N = 1000000;
flux = 2.0;

y = quanta_sample_direct(flux * ones(1, N), [1, N], inf, 0);
fprintf('Poisson (M=inf), flux=%g: mean=%.4f (theory %.4f), var=%.4f (theory %.4f)\n', ...
    flux, mean(y), flux, var(y), flux);

y = quanta_sample_direct(flux * ones(1, N), [1, N], 1, 0);
p = ppd(flux);
fprintf('Binary  (M=1),   flux=%g: mean=%.4f (theory %.4f), var=%.4f (theory %.4f)\n', ...
    flux, mean(y), p, var(y), p*(1-p));

y = quanta_sample_direct(flux * ones(1, N), [1, N], 4, 0.5);
fprintf('M=4, read=0.5,   flux=%g: mean=%.4f, var=%.4f\n', flux, mean(y), var(y));

fprintf('\n=== quanta_sample_burst (statistical) ===\n');
flux = 4.0;
M = 8;
y = quanta_sample_burst(flux * ones(1, N), [1, N], M, 0);
p = ppd(flux/M);
mean_th = M * p;
var_th = M * p * (1 - p);
fprintf('burst  M=%d, flux=%g: mean=%.4f (theory %.4f), var=%.4f (theory %.4f)\n', ...
    M, flux, mean(y), mean_th, var(y), var_th);

y2 = quanta_sample_direct(flux * ones(1, N), [1, N], M, 0);
fprintf('direct M=%d, flux=%g: mean=%.4f (theory %.4f), var=%.4f (theory %.4f)\n', ...
    M, flux, mean(y2), flux, var(y2), flux);