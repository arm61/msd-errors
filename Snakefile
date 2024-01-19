rule mv_schematic:
    input:
        "src/static/schematic.pdf"
    output:
        "src/tex/figures/schematic.pdf"
    shell:
        "cp {input} {output}"

rule glswlsols:
    input:
        "src/code/random_walks/glswlsols.py",
        "src/data/random_walks/numerical/rw_1_128_128_s4096.npz"
    output:
        "src/data/random_walks/numerical/glswlsols_1_128_128.npz"
    conda:
        "environment.yml"
    cache: 
        True
    params:
        jump=2.4494897428,
        atoms=128,
        length=128,
        correlation="true"
    script:
        "src/code/random_walks/glswlsols.py"

rule stat_eff_generation:
    input: 
        'src/scripts/stat_eff.py',
        'src/data/random_walks/stat_eff.npz',
        'src/scripts/utils/plotting_helper.py',
        'src/scripts/utils/_fig_params.py'
    output:
        'src/tex/figures/stat_eff.pdf'
    conda:
        'environment.yml'
    shell:
        'cd src/scripts && python stat_eff.py'

rule:
    name:
        f"stat_eff_true"
    input:
        'src/code/random_walks/stat_eff.py',
        [f'src/data/random_walks/kinisi/rw_1_128_{length}_s512.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/weighted/rw_1_128_{length}_s512.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/ordinary/rw_1_128_{length}_s512.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/numerical/D_1_128_{length}.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/kinisi/rw_1_{atoms}_128_s512.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/weighted/rw_1_{atoms}_128_s512.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/ordinary/rw_1_{atoms}_128_s512.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/numerical/D_1_{atoms}_128.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]]
    output:
        f'src/data/random_walks/stat_eff.npz'
    conda: 
        'environment.yml'
    cache: 
        True
    params:
        correlation='true'
    priority: 50
    script:
        "src/code/random_walks/stat_eff.py"

rule efficiency:
    input:
        'src/code/random_walks/efficiency.py',
        [f'src/data/random_walks/kinisi/rw_1_128_{length}_s512.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/weighted/rw_1_128_{length}_s512.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/ordinary/rw_1_128_{length}_s512.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/numerical/D_1_128_{length}.npz' for length in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/kinisi/rw_1_{atoms}_128_s512.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/weighted/rw_1_{atoms}_128_s512.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/ordinary/rw_1_{atoms}_128_s512.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]],
        [f'src/data/random_walks/numerical/D_1_{atoms}_128.npz' for atoms in [16, 32, 64, 128, 256, 512, 1024]]
    output:
        f'src/data/random_walks/efficiency.npz'
    conda: 
        'environment.yml'
    cache: 
        True
    params:
        correlation='true'
    script:
        "src/code/random_walks/efficiency.py"

# Random walks simulations
rule rw_4096_kinisi_true_128:
    input:
        f'src/code/random_walks/kinisi_rw.py',
        'src/code/random_walks/random_walk.py'
    output:
        f'src/data/random_walks/kinisi/rw_1_128_128_s4096.npz'
    conda:
        'environment.yml'
    cache:
        True
    params:
        jump=2.4494897428,
        atoms=128,
        length=128,
        correlation='true',
        n=4096
    script:
        f'src/code/random_walks/kinisi_rw.py'

# Random walks simulations
rule rw_4096_pyblock_true_128:
    input:
        f'src/code/random_walks/kinisi_pyblock_rw.py',
        'src/code/random_walks/random_walk.py'
    output:
        f'src/data/random_walks/pyblock/rw_1_128_128_s4096.npz'
    conda:
        'environment.yml'
    cache:
        True
    params:
        jump=2.4494897428,
        atoms=128,
        length=128,
        correlation='true',
        n=4096
    script:
        f'src/code/random_walks/kinisi_pyblock_rw.py'

# Random walks simulations
rule rw_4096_pyblock_modelfree_true_128:
    input:
        f'src/code/random_walks/kinisi_pyblock_modelfree_rw.py',
        'src/code/random_walks/random_walk.py'
    output:
        f'src/data/random_walks/pyblock_modelfree/rw_1_128_128_s4096.npz'
    conda:
        'environment.yml'
    cache:
        True
    params:
        jump=2.4494897428,
        atoms=128,
        length=128,
        correlation='true',
        n=4096
    script:
        f'src/code/random_walks/kinisi_pyblock_modelfree_rw.py'

# Length variation random walks 
for length in [16, 32, 64, 128, 256, 512, 1024]:
    rule:
        name:
            f"rw_4096_numerical_length_true_{length}"
        input:
            'src/code/random_walks/numerical_rw.py',
            'src/code/random_walks/random_walk.py'
        output:
            f'src/data/random_walks/numerical/rw_1_128_{length}_s4096.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=128,
            length=length,
            correlation='true',
            n=4096
        script:
            "src/code/random_walks/numerical_rw.py"

for length in [16, 32, 64, 128, 256, 512, 1024]:
    rule:
        name:
            f"true_D_length_{length}_true"
        input:
            'src/code/random_walks/true_ls.py',
            f'src/data/random_walks/numerical/rw_1_128_{length}_s4096.npz'
        output:
            f'src/data/random_walks/numerical/D_1_128_{length}.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            atoms=128,
            jump=2.4494897428,
            length=length,
            correlation='true',
        script:
            "src/code/random_walks/true_ls.py"
   
for length in [16, 32, 64, 256, 512, 1024]: 
    rule:
        name:
            f'rw_512_length_{length}_kinisi_true'
        input:
            f'src/code/random_walks/kinisi_rw.py',
            'src/code/random_walks/random_walk.py',
        output:
            f'src/data/random_walks/kinisi/rw_1_128_{length}_s512.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=128,
            length=length,
            correlation='true',
            n=512 
        script:
            f'src/code/random_walks/kinisi_rw.py'

for length in [16, 32, 64, 256, 512, 1024]: 
    rule:
        name:
            f'rw_512_length_{length}_weighted_true'
        input:
            f'src/code/random_walks/weighted_rw.py',
            'src/code/random_walks/random_walk.py',
            f'src/data/random_walks/kinisi/rw_1_128_{length}_s512.npz'
        output:
            f'src/data/random_walks/weighted/rw_1_128_{length}_s512.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=128,
            length=length,
            correlation='true',
            n=512 
        script:
            f'src/code/random_walks/weighted_rw.py'

for length in [16, 32, 64, 256, 512, 1024]: 
    rule:
        name:
            f'rw_512_length_{length}_ordinary_true'
        input:
            f'src/code/random_walks/ordinary_rw.py',
            'src/code/random_walks/random_walk.py',
            f'src/data/random_walks/kinisi/rw_1_128_{length}_s512.npz'
        output:
            f'src/data/random_walks/ordinary/rw_1_128_{length}_s512.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=128,
            length=length,
            correlation='true',
            n=512 
        script:
            f'src/code/random_walks/ordinary_rw.py'

# Atoms variation random walks
for atoms in [2, 4, 16, 32, 64, 256, 512, 1024]:
    rule:
        name:
            f"rw_4096_numerical_atoms_true_{atoms}"
        input:
            'src/code/random_walks/numerical_rw.py',
            'src/code/random_walks/random_walk.py'
        output:
            f'src/data/random_walks/numerical/rw_1_{atoms}_128_s4096.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=atoms,
            length=128,
            correlation='true',
            n=4096
        script:
            "src/code/random_walks/numerical_rw.py"

for atoms in [16, 32, 64, 256, 512, 1024]:
    rule:
        name:
            f"true_D_atoms_{atoms}_true"
        input:
            'src/code/random_walks/true_ls.py',
            f'src/data/random_walks/numerical/rw_1_{atoms}_128_s4096.npz'
        output:
            f'src/data/random_walks/numerical/D_1_{atoms}_128.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            atoms=atoms,
            jump=2.4494897428,
            length=128,
            correlation='true',
        script:
            "src/code/random_walks/true_ls.py"

for atoms in [16, 32, 64, 128, 256, 512, 1024]: 
    rule:
        name:
            f'rw_512_atoms_{atoms}_kinisi_true'
        input:
            'src/code/random_walks/kinisi_rw.py',
            'src/code/random_walks/random_walk.py'
        output:
            f'src/data/random_walks/kinisi/rw_1_{atoms}_128_s512.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=atoms,
            length=128,
            correlation='true',
            n=512 
        script:
            f'src/code/random_walks/kinisi_rw.py'

for atoms in [16, 32, 64, 128, 256, 512, 1024]: 
    rule:
        name:
            f'rw_512_atoms_{atoms}_weighted_true'
        input:
            f'src/code/random_walks/weighted_rw.py',
            'src/code/random_walks/random_walk.py',
            f'src/data/random_walks/kinisi/rw_1_{atoms}_128_s512.npz'
        output:
            f'src/data/random_walks/weighted/rw_1_{atoms}_128_s512.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=atoms,
            length=128,
            correlation='true',
            n=512 
        script:
            f'src/code/random_walks/weighted_rw.py'

for atoms in [16, 32, 64, 128, 256, 512, 1024]: 
    rule:
        name:
            f'rw_512_atoms_{atoms}_ordinary_true'
        input:
            f'src/code/random_walks/ordinary_rw.py',
            'src/code/random_walks/random_walk.py',
            f'src/data/random_walks/kinisi/rw_1_{atoms}_128_s512.npz'
        output:
            f'src/data/random_walks/ordinary/rw_1_{atoms}_128_s512.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            jump=2.4494897428,
            atoms=atoms,
            length=128,
            correlation='true',
            n=512 
        script:
            f'src/code/random_walks/ordinary_rw.py'

# True covariance matrix analysis
rule rw_4096_true_cov_true_128:
    input:
        f'src/data/random_walks/kinisi/rw_1_128_128_s4096.npz',
        f'src/data/random_walks/numerical/rw_1_128_128_s4096.npz',
        'src/code/random_walks/random_walk.py'
    output:
        f'src/data/random_walks/true_cov/rw_1_128_128_s4096.npz'
    conda:
        'environment.yml'
    cache:
        True
    params:
        jump=2.4494897428,
        atoms=128,
        length=128,
        correlation='true',
        n=4096
    script:
        f'src/code/random_walks/truecov_rw.py'

for start_diff in [0, 2, 4, 6, 8, 10, 15, 20]:
    for n in range(0, 6, 1):
        rule:
            name:
                f"llzo_many_{n}_{start_diff}"
            input:
                'src/code/llzo/many_runs.py',
                # f'src/data/llzo/traj{n}.xyz'
            output:
                f'src/data/llzo/diffusion_{n}_{start_diff}.npz'

            conda:
                'environment.yml'
            cache:
                True
            params:
                n=n,
                start_diff=start_diff
            script:
                "src/code/llzo/many_runs.py"

    rule:
        name:
            f"true_D_llzo_{start_diff}"
        input:
            'src/code/llzo/true_ls.py',
            [f'src/data/llzo/diffusion_{n}_{start_diff}.npz' for n in range(0, 6, 1)]
        output:
            f'src/data/llzo/true_{start_diff}.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            start_diff=start_diff
        script:
            "src/code/llzo/true_ls.py"

    rule:
        name:
            f"glswlsols_D_llzo_{start_diff}"
        input:
            'src/code/llzo/glswlsols.py',
            [f'src/data/llzo/diffusion_{n}_{start_diff}.npz' for n in range(0, 6, 1)]
        output:
            f'src/data/llzo/glswlsols_{start_diff}.npz'
        conda:
            'environment.yml'
        cache:
            True
        params:
            start_diff=start_diff
        script:
            "src/code/llzo/glswlsols.py"
        