import React, { Component } from 'react';
import {ReactCytoscape} from 'react-cytoscape';
import cytoscape from 'cytoscape';
import './style.css'

class Graph extends Component {

	getElements() {
		return {
			nodes: [
				{ data: { id: 'a'}, position: { x: 215, y: 85 } },
				{ data: { id: 'b' } },
				{ data: { id: 'c'}, position: { x: 300, y: 85 } },
				{ data: { id: 'd' }, position: { x: 215, y: 175 } },
				{ data: { id: 'e' } },
				{ data: { id: 'f'}, position: { x: 300, y: 175 } }
			],
			edges: [
				{ data: { id: 'ad', source: 'a', target: 'd' } },
				{ data: { id: 'eb', source: 'e', target: 'b' } }
			]
		};
	}

	render() {

		return (
		<ReactCytoscape containerID="cy"
			elements={this.props.elements.elements}
			cyRef={(cy) => { this.cyRef(cy) }}
			style = {[
				{
				  selector: 'node',
				  style: {
					'label': 'data(label)',
					'width': 'mapData(pc_degree,0,0.07,10,400)',
					'height': 'mapData(pc_degree,0,0.07,10,400)'
				  }
				},
				{
					selector: "node[pred_subject = 'Neural_Networks']",
					style: {
					  'background-color': '#00416A'
					}
				},
				{
					selector: "node[pred_subject = 'Rule_Learning']",
					style: {
					  'background-color': '#008000'
					}
				},
				{
					selector: "node[pred_subject = 'Reinforcement_Learning']",
					style: {
					  'background-color': '#F09511'
					}
				},
				{
					selector: "node[pred_subject = 'Probabilistic_Methods']",
					style: {
					  'background-color': '#F09511'
					}
				},
				{
					selector: "node[pred_subject = 'Theory']",
					style: {
					  'background-color': '#A00000'
					}
				},
				{
					selector: "node[pred_subject = 'Genetic_Algorithms']",
					style: {
					  'background-color': '#222222'
					}
				},
				{
					selector: "node[pred_subject = 'Case_Based']",
					style: {
					  'background-color': '#1460AA'
					}
				}
			]}
			cytoscapeOptions={{}}
			layout={{ name: 'cose' }} />
		);
	}

	cyRef(cy) {
		this.cy = cy;
		cy.on('tap', 'node', function (evt) {
			var node = evt.target;
			console.log('tapped ' + node.id());
		});
	}

	handleEval() {
		const cy = this.cy;
		const str = this.text.value;
		eval(str);
	}
}

export default Graph;