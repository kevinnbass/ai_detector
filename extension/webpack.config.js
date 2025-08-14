const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  
  return {
    mode: isProduction ? 'production' : 'development',
    devtool: isProduction ? false : 'source-map',
    
    entry: {
      'background/background': './src/background/background.js',
      'background/gemini-analyzer': './src/background/gemini-analyzer.js',
      'content/content': './src/content/content.js',
      'content/detector-engine': './src/content/detector-engine.js',
      'content/llm-integration': './src/content/llm-integration.js',
      'popup/popup': './src/popup/popup.js'
    },
    
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: '[name].js',
      clean: true
    },
    
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: ['@babel/preset-env'],
              plugins: ['@babel/plugin-transform-runtime']
            }
          }
        },
        {
          test: /\.css$/,
          use: [MiniCssExtractPlugin.loader, 'css-loader']
        },
        {
          test: /\.(png|jpg|gif|svg)$/,
          type: 'asset/resource',
          generator: {
            filename: 'assets/[name][ext]'
          }
        }
      ]
    },
    
    plugins: [
      new CleanWebpackPlugin(),
      
      new MiniCssExtractPlugin({
        filename: '[name].css'
      }),
      
      new CopyPlugin({
        patterns: [
          { from: 'src/manifest.json', to: 'manifest.json' },
          { from: 'src/popup/*.html', to: 'popup/[name][ext]' },
          { from: 'assets/icons', to: 'icons' },
          { from: 'src/content/detection-rules.json', to: 'content/detection-rules.json' }
        ]
      })
    ],
    
    optimization: {
      minimize: isProduction,
      minimizer: [
        new TerserPlugin({
          terserOptions: {
            compress: {
              drop_console: isProduction,
              drop_debugger: isProduction
            },
            format: {
              comments: false
            }
          },
          extractComments: false
        }),
        new CssMinimizerPlugin()
      ],
      splitChunks: {
        chunks: 'async',
        cacheGroups: {
          shared: {
            name: 'shared/common',
            chunks: 'all',
            minChunks: 2,
            priority: 10,
            reuseExistingChunk: true,
            enforce: true
          }
        }
      }
    },
    
    performance: {
      hints: isProduction ? 'warning' : false,
      maxEntrypointSize: 512000,
      maxAssetSize: 512000
    },
    
    stats: {
      colors: true,
      modules: false,
      children: false,
      chunks: false,
      chunkModules: false
    }
  };
};