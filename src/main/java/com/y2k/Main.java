package com.y2k;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.converters.ConverterUtils.DataSource;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class Main {
    private J48 tree;
    private Instances dataset;

    public Main() {
        try {
            DataSource dataSource = new DataSource("src/main/resources/breast-cancer.arff");
            dataset = dataSource.getDataSet();
            dataset.setClassIndex(dataset.numAttributes() - 1);

            tree = new J48();
            tree.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void createAndShowGUI() {
        JFrame frame = new JFrame("Clasificador WEKA");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());

        JLabel title = new JLabel("Clasificador WEKA", SwingConstants.CENTER);
        title.setFont(new Font("Arial", Font.BOLD, 24));
        title.setForeground(new Color(34, 45, 65));
        title.setBorder(BorderFactory.createEmptyBorder(20, 10, 20, 10));
        frame.add(title, BorderLayout.NORTH);

        JPanel inputPanel = new JPanel(new GridLayout(0, 2, 10, 10));
        inputPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        inputPanel.setBackground(new Color(245, 245, 245));

        JTextField[] inputFields = new JTextField[dataset.numAttributes() - 1];
        for (int i = 0; i < dataset.numAttributes() - 1; i++) {
            JLabel label = new JLabel(dataset.attribute(i).name() + ":");
            label.setFont(new Font("Arial", Font.PLAIN, 16));
            inputFields[i] = new JTextField();
            inputFields[i].setFont(new Font("Arial", Font.PLAIN, 16));
            inputPanel.add(label);
            inputPanel.add(inputFields[i]);
        }
        frame.add(inputPanel, BorderLayout.CENTER);

        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        JButton classifyButton = new JButton("Clasificar");
        classifyButton.setFont(new Font("Arial", Font.BOLD, 16));
        classifyButton.setBackground(new Color(34, 140, 60));
        classifyButton.setForeground(Color.WHITE);
        classifyButton.setFocusPainted(false);

        JLabel resultLabel = new JLabel("Resultado: ", SwingConstants.CENTER);
        resultLabel.setFont(new Font("Arial", Font.BOLD, 18));
        resultLabel.setForeground(new Color(34, 45, 65));
        resultLabel.setBorder(BorderFactory.createEmptyBorder(20, 10, 20, 10));

        classifyButton.addActionListener((ActionEvent e) -> {
            try {
                double[] values = new double[dataset.numAttributes()];
                for (int i = 0; i < inputFields.length; i++) {
                    String userInput = inputFields[i].getText();

                    if (userInput.isEmpty()) {
                        if (dataset.attribute(i).isNumeric()) {
                            values[i] = dataset.meanOrMode(i);
                        } else {
                            values[i] = dataset.meanOrMode(i);
                        }
                    } else {
                        if (dataset.attribute(i).isNumeric()) {
                            values[i] = Double.parseDouble(userInput);
                        } else {
                            values[i] = dataset.attribute(i).indexOfValue(userInput);
                        }
                    }
                }
                values[dataset.classIndex()] = Double.NaN;

                DenseInstance newInstance = new DenseInstance(1.0, values);
                newInstance.setDataset(dataset);

                double predictedClass = tree.classifyInstance(newInstance);
                String className = dataset.classAttribute().value((int) predictedClass);

                resultLabel.setText("Resultado: " + className);
            } catch (Exception ex) {
                ex.printStackTrace();
                resultLabel.setText("Error: " + ex.getMessage());
            }
        });

        bottomPanel.add(classifyButton, BorderLayout.NORTH);
        bottomPanel.add(resultLabel, BorderLayout.SOUTH);
        frame.add(bottomPanel, BorderLayout.SOUTH);

        frame.getContentPane().setBackground(new Color(245, 245, 245));

        frame.pack();

        frame.setLocationRelativeTo(null);

        frame.setVisible(true);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            Main classifier = new Main();
            classifier.createAndShowGUI();
        });
    }
}
