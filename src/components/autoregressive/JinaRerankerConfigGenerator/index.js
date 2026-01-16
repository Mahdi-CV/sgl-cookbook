import React from 'react';
import ConfigGenerator from '../../base/ConfigGenerator';

/**
 * Jina Reranker m0 Configuration Generator
 * Supports AMD GPUs (MI300X, MI325X, MI355X)
 */
const JinaRerankerConfigGenerator = () => {
  const config = {
    modelFamily: 'jinaai',

    options: {
      hardware: {
        name: 'hardware',
        title: 'Hardware Platform',
        items: [
          { id: 'mi300x', label: 'MI300X', default: true },
          { id: 'mi325x', label: 'MI325X', default: false },
          { id: 'mi355x', label: 'MI355X', default: false }
        ]
      }
    },

    generateCommand: function(values) {
      const { hardware } = values;

      let cmd = 'python3 -m sglang.launch_server \\\\\n';
      cmd += '  --model-path jinaai/jina-reranker-m0 \\\\\n';
      cmd += '  --tp 1 \\\\\n';
      cmd += '  --trust-remote-code --is-embedding \\\\\n';
      cmd += '  --disable-radix-cache --attention-backend triton --skip-server-warmup';

      return cmd;
    }
  };

  return <ConfigGenerator config={config} />;
};

export default JinaRerankerConfigGenerator;
